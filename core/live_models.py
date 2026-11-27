"""Client and framed pipe protocol for the isolated live inference runtime."""

from __future__ import annotations

from contextlib import suppress
import json
import os
from pathlib import Path
from queue import Empty, Queue
import struct
import subprocess
import sys
from threading import Lock, Thread
from typing import TYPE_CHECKING

from core.ocr_settings import FLOW_DIRECT, FLOW_NLLB_VINAI, OCRSettings

if TYPE_CHECKING:
    import numpy as np

_HEADER = struct.Struct("!II")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_FRAME_BYTES = 128 * 1024 * 1024
_WORKER = Path(__file__).with_name("live_inference_worker.py").resolve()
_ROOT = _WORKER.parent.parent


class LiveModelError(RuntimeError):
    """The live inference worker could not complete a request."""


def read_packet(stream):
    header = stream.read(_HEADER.size)
    if not header:
        return None
    if len(header) != _HEADER.size:
        raise ValueError("Truncated live model packet header")
    json_size, binary_size = _HEADER.unpack(header)
    if not 0 < json_size <= _MAX_JSON_BYTES or binary_size > _MAX_FRAME_BYTES:
        raise ValueError("Live model packet exceeds size limits")
    json_data = stream.read(json_size)
    if len(json_data) != json_size:
        raise ValueError("Truncated live model packet metadata")
    metadata = json.loads(json_data)
    if not isinstance(metadata, dict):
        raise ValueError("Invalid live model packet metadata")
    binary = stream.read(binary_size)
    if len(binary) != binary_size:
        raise ValueError("Truncated live model packet data")
    return metadata, binary


def write_packet(stream, metadata, binary=b""):
    json_data = json.dumps(metadata, ensure_ascii=True).encode("ascii")
    if len(json_data) > _MAX_JSON_BYTES or len(binary) > _MAX_FRAME_BYTES:
        raise ValueError("Live model packet exceeds size limits")
    stream.write(_HEADER.pack(len(json_data), len(binary)))
    stream.write(json_data)
    if binary:
        stream.write(binary)
    stream.flush()


def _worker_environment():
    env = os.environ.copy()
    # The worker resolves model directories itself before creating PaddleOCR.
    env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    # Changing OCR tensor shapes accumulate cuDNN convolution plans in host RAM.
    env.setdefault("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT", "512")
    runtime = env.get("LIVE_PYTHONPATH")
    default_runtime = _ROOT / "models" / "live-runtime"
    if runtime is None and default_runtime.is_dir():
        runtime = str(default_runtime)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (runtime, str(_ROOT), env.get("PYTHONPATH")) if path
    )
    return env


def check_qwen_cuda_worker() -> None:
    """Check PyTorch in the live worker environment without loading a model."""
    env = _worker_environment()
    executable = env.get("MANGALENS_LIVE_PYTHON") or sys.executable
    probe = (
        "try:\n"
        "    from core.live_qwen_cuda import check_cuda_torch\n"
        "    check_cuda_torch()\n"
        "except Exception as error:\n"
        "    raise SystemExit(str(error))"
    )
    try:
        result = subprocess.run(
            [executable, "-c", probe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, cwd=_ROOT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            encoding="utf-8", errors="replace", timeout=10,
        )
    except subprocess.TimeoutExpired as error:
        raise LiveModelError("The CUDA compatibility check timed out.") from error
    except OSError as error:
        raise LiveModelError(f"Could not start the live inference worker. {error}") from error
    if result.returncode:
        lines = result.stdout.strip().splitlines()
        raise LiveModelError(lines[-1] if lines else "The CUDA compatibility check failed.")


class LiveModelClient:
    """Run OCR and translation synchronously on one inference thread.

    The process starts on the first nonempty request. Another thread may call
    close() to interrupt a pending request and release the child process.
    """

    def __init__(self, *, request_timeout: float = 180.0, ocr_settings: OCRSettings = OCRSettings()):
        self._request_timeout = request_timeout
        self._ocr_settings = ocr_settings
        self._lock = Lock()
        self._process = None
        self._closed = False
        self.last_token_count: int | None = None

    def ocr(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], str]]:
        import numpy as np

        if (
            not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.dtype != np.uint8
            or frame.size == 0
        ):
            raise ValueError("Live OCR requires a nonempty uint8 BGR image")
        if frame.nbytes > _MAX_FRAME_BYTES:
            raise ValueError("Live OCR frame exceeds the size limit")
        height, width = frame.shape[:2]
        binary = memoryview(np.ascontiguousarray(frame)).cast("B")
        rows = self._request({
            "op": "ocr", "height": height, "width": width,
            "det": self._ocr_settings.det, "rec": self._ocr_settings.rec,
        }, binary)
        if not isinstance(rows, list):
            self._invalid_response("invalid OCR result")
        result = []
        for row in rows:
            if not isinstance(row, list) or len(row) != 2:
                self._invalid_response("invalid OCR entry")
            box, text = row
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(type(value) is not int for value in box)
                or not isinstance(text, str)
            ):
                self._invalid_response("invalid OCR box or text")
            left, top, right, bottom = box
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                self._invalid_response("OCR box is outside the frame")
            result.append(((left, top, right, bottom), text))
        return result

    def translate(self, texts: list[str]) -> list[str]:
        return self._translate(texts)

    def translate_contextual(self, text: str, *, role: str, rules: str, speaker: str, history) -> str:
        context = {"role": role, "rules": rules, "speaker": speaker, "history": history}
        return self._translate([text], context)[0]

    def _translate(self, texts: list[str], context=None) -> list[str]:
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise TypeError("Live translation expects a list of strings")
        self.last_token_count = None
        if not texts:
            with self._lock:
                if self._closed:
                    raise LiveModelError("Live model client is closed")
                return []
        metadata = {"op": "translate", "texts": texts, "ja_model": self._ocr_settings.ja_model}
        if self._ocr_settings.flow == FLOW_NLLB_VINAI:
            metadata["flow"] = FLOW_NLLB_VINAI
        if self._ocr_settings.flow == FLOW_DIRECT:
            metadata.update(
                flow=FLOW_DIRECT, qwen_context=self._ocr_settings.qwen_context,
                qwen_model=self._ocr_settings.qwen_model, qwen_backend=self._ocr_settings.qwen_backend,
            )
            if self._ocr_settings.qwen_rule is not None:
                metadata["qwen_rules"] = self._ocr_settings.qwen_rule.text
            if context is not None:
                metadata["context"] = context
        result = self._request(metadata)
        if not _is_translation_batch(result, len(texts)):
            self._invalid_response("the translator must return one string per input")
        return result

    def _get_process(self):
        with self._lock:
            if self._closed:
                raise LiveModelError("Live model client is closed")
            if self._process is None:
                env = _worker_environment()
                executable = env.get("MANGALENS_LIVE_PYTHON") or sys.executable
                self._process = subprocess.Popen(
                    [executable, "-u", str(_WORKER)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL if sys.stderr is None else None,
                    env=env,
                    cwd=_ROOT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            return self._process

    def _request(self, metadata, binary=b""):
        operation = metadata["op"]
        try:
            process = self._get_process()
        except OSError as error:
            self.close()
            raise LiveModelError(f"Could not start the live inference worker: {error}") from error

        # Timed pipe I/O also covers a worker that stops reading stdin mid-frame.
        reply = Queue(maxsize=1)

        def exchange():
            try:
                write_packet(process.stdin, metadata, binary)
                reply.put(read_packet(process.stdout))
            except (OSError, ValueError) as error:
                reply.put(error)

        thread = Thread(target=exchange, name="MangaLens live model pipe", daemon=True)
        thread.start()
        try:
            response = reply.get(timeout=self._request_timeout)
        except Empty:
            self.close()
            raise LiveModelError(f"Live {operation} timed out") from None
        if isinstance(response, Exception):
            self.close()
            raise LiveModelError(f"Live {operation} connection failed: {response}") from response
        if response is None:
            self.close()
            raise LiveModelError(
                f"Live inference worker exited during {operation}. Check the live runtime setup."
            )
        message, data = response
        if message.get("ok") is False and isinstance(message.get("error"), str) and not data:
            self.close()
            raise LiveModelError(f"Live {operation} failed: {message['error']}")
        if message.get("ok") is not True or "result" not in message or data:
            self._invalid_response("invalid response packet")
        if operation == "translate" and self._ocr_settings.flow == FLOW_DIRECT:
            count = message.get("token_count")
            if type(count) is int and count >= 0:
                self.last_token_count = count
        return message["result"]

    def _invalid_response(self, reason):
        self.close()
        raise LiveModelError(f"Live inference protocol error: {reason}")

    def close(self):
        """Terminate and reap the worker. Safe before startup or after failure."""
        with self._lock:
            self._closed = True
            process, self._process = self._process, None
        if process is None:
            return
        # Reap the child before closing the streams, since a background I/O
        # thread may be waiting on either pipe while holding its buffer lock.
        if process.poll() is None:
            process.kill()
        process.wait()
        for stream in (process.stdin, process.stdout):
            with suppress(OSError):
                stream.close()


def _is_translation_batch(result, size):
    return (
        isinstance(result, list)
        and len(result) == size
        and all(isinstance(text, str) for text in result)
    )
