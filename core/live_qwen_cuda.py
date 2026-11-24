"""Local Qwen inference through the optional Windows CUDA Python runtime."""

from contextlib import ExitStack
import ctypes
import os
from pathlib import Path
import sys

from core.live_qwen import QwenContextLimitError, QwenTranslator


def check_cuda_torch() -> Path:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("llama-cpp-python CUDA needs an NVIDIA GPU and CUDA-enabled PyTorch.")
    torch_dir = Path(torch.__file__).resolve().parent / "lib"
    if not all((torch_dir / name).is_file() for name in ("cublas64_13.dll", "cublasLt64_13.dll")):
        raise RuntimeError("Install a PyTorch build with CUDA 13 libraries for this backend.")
    return torch_dir


def _open_cuda_model(model_path: Path, runtime: Path, context_size: int):
    package_dir = runtime / "python"
    native_dir = package_dir / "llama_cpp" / "lib"
    native_library = native_dir / "llama.dll"
    if not native_library.is_file():
        raise RuntimeError("Install llama-cpp-python CUDA in Setup before using this backend.")

    torch_dir = check_cuda_torch()

    with ExitStack() as libraries:
        libraries.enter_context(os.add_dll_directory(str(native_dir)))
        libraries.enter_context(os.add_dll_directory(str(torch_dir)))
        # Preload using Windows' secure DLL search so the wheel can resolve PyTorch's cuBLAS.
        ctypes.WinDLL(str(native_library))
        sys.path.insert(0, str(package_dir))
        previous = os.environ.get("LLAMA_CPP_LIB_PATH")
        os.environ["LLAMA_CPP_LIB_PATH"] = str(native_dir)
        try:
            from llama_cpp import Llama, llama_supports_gpu_offload
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        finally:
            sys.path.remove(str(package_dir))
            if previous is None:
                os.environ.pop("LLAMA_CPP_LIB_PATH", None)
            else:
                os.environ["LLAMA_CPP_LIB_PATH"] = previous
        if not llama_supports_gpu_offload():
            raise RuntimeError("The installed llama-cpp-python runtime does not support GPU offload.")
        model = Llama(
            model_path=str(model_path), n_ctx=context_size, n_gpu_layers=-1,
            main_gpu=0, flash_attn=True, verbose=False,
        )
    try:
        bos, eos = model.token_bos(), model.token_eos()
        formatter = Jinja2ChatFormatter(
            template=model.metadata["tokenizer.chat_template"],
            bos_token=model.detokenize([bos], special=True).decode("utf-8") if bos >= 0 else "",
            eos_token=model.detokenize([eos], special=True).decode("utf-8"),
        )
    except Exception:
        model.close()
        raise
    return model, formatter


class PythonCudaQwenTranslator(QwenTranslator):
    """Reuse Qwen prompts and validation with in-process CUDA inference."""

    def __init__(self, model_path: Path, runtime: Path, context_size: int, *, rules: str = ""):
        super().__init__(model_path, None, context_size, rules=rules)
        self._runtime = runtime.resolve()
        self._model = None
        self._formatter = None

    def _start(self):
        if self._model is not None:
            return
        if not self._model_path.is_file():
            raise RuntimeError("Install the selected Qwen model in Setup.")
        try:
            self._model, self._formatter = _open_cuda_model(self._model_path, self._runtime, self._context_size)
        except (ImportError, OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"Could not start Qwen on CUDA. Check Setup, PyTorch CUDA and available VRAM, "
                f"or select Vulkan in Advanced settings. {error}"
            ) from error

    def _complete(self, messages, max_tokens):
        formatted = self._formatter(messages=messages, enable_thinking=False)
        tokens = self._model.tokenize(formatted.prompt.encode("utf-8"), add_bos=False, special=True)
        if len(tokens) + max_tokens > self._model.n_ctx():
            raise QwenContextLimitError("The Qwen prompt exceeds the CUDA context. Increase the context in Settings.")
        result = self._model.create_completion(
            tokens, max_tokens=max_tokens, temperature=0, stop=formatted.stop, stream=False,
        )
        choice = result["choices"][0]
        return {
            "choices": [{
                "message": {"content": choice["text"]},
                "finish_reason": choice.get("finish_reason"),
            }],
            "usage": result.get("usage"),
        }

    def close(self):
        model, self._model = self._model, None
        self._formatter = None
        if model is not None:
            model.close()
