"""Translate English or Japanese OCR lines into Vietnamese."""

import ctypes
import json
import logging
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from unicodedata import normalize

from core.ocr_settings import DEFAULT_JA_MODEL, JA_MODEL_KEYS
from core.setup_models import NLLB_CUSTOM_MODEL_KEY, NLLB_ONNX_FILES, _file_present, _index_shards, resolve_model

_ROOT = Path(__file__).resolve().parents[1]
_BATCH_SIZE = 32
_MAX_NEW_TOKENS = 96
_CT2_SOURCE = "mangalens-source.json"
_LOG = logging.getLogger(__name__)
_LATIN_SCRIPT = re.compile(r"[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f]")
_JAPANESE_SCRIPT = re.compile(
    r"[\u3005\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fd-\u30ff"
    r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9d"
    r"\U00020000-\U0002fa1f]"
)


def _from_pretrained(loader, source, **kwargs):
    # Local cache hits should work without waiting for a remote metadata check.
    try:
        return loader.from_pretrained(source, local_files_only=True, **kwargs)
    except OSError:
        return loader.from_pretrained(source, **kwargs)


def _has_indexed_safetensors(checkpoint: Path) -> bool:
    index = checkpoint / "model.safetensors.index.json"
    shards = _index_shards(index)
    if not shards:
        return False
    try:
        weight_map = json.loads(index.read_bytes())["weight_map"]
        for shard in shards:
            path = checkpoint / shard
            if path.suffix != ".safetensors" or not _file_present(path):
                return False
            with path.open("rb") as stream:
                header_size = int.from_bytes(stream.read(8), "little")
                if not 0 < header_size <= 100_000_000:
                    return False
                header = json.loads(stream.read(header_size))
            weights = {weight for weight, name in weight_map.items() if name == shard}
            if weights != header.keys() - {"__metadata__"}:
                return False
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return False
    return True


def _local_fp16_checkpoint(directory: str, model_id: str, tokenizer_id: str) -> Path | None:
    from transformers.utils import is_accelerate_available

    if not is_accelerate_available():
        return None
    checkpoint = _ROOT / directory
    if not _file_present(checkpoint / "config.json"):
        return None
    try:
        source = json.loads((checkpoint / "mangalens-source.json").read_bytes())
    except (OSError, ValueError):
        return None
    if (
        not isinstance(source, dict)
        or source.get("model") != model_id
        or source.get("tokenizer") != tokenizer_id
    ):
        return None
    return checkpoint if _has_indexed_safetensors(checkpoint) else None


def _load_cuda_blas() -> bool:
    """CTranslate2 links CUDA 12 cuBLAS, which PyTorch CUDA 13 builds do not include."""
    try:
        from nvidia import cublas

        directory = Path(next(iter(cublas.__path__))) / "bin"
    except (ImportError, StopIteration):
        directory = None
    try:
        for name in ("cublasLt64_12.dll", "cublas64_12.dll"):
            ctypes.WinDLL(str(directory / name) if directory else name)
    except OSError:
        return False
    return True


def _convert_ctranslate2(model_id, tokenizer, output: Path) -> int:
    from ctranslate2.converters import transformers as converters
    import torch
    from transformers import AutoModelForSeq2SeqLM

    class MarianLoader(converters.MarianMTLoader):
        # Transformers starts the decoder from the <pad> embedding, which is nonzero in
        # Mitsua ElanMT. Keep that token instead of CTranslate2's zero-embedding start.
        def get_model_spec(self, model):
            model.config.normalize_before = False
            model.config.normalize_embedding = False
            return converters.BartLoader.get_model_spec(self, model)

        def set_decoder(self, spec, decoder):
            converters.BartLoader.set_decoder(self, spec, decoder)

        def get_vocabulary(self, model, tokenizer):
            return converters.BartLoader.get_vocabulary(self, model, tokenizer)

        def set_config(self, config, model, tokenizer):
            super().set_config(config, model, tokenizer)
            config.decoder_start_token = tokenizer.convert_ids_to_tokens(model.config.decoder_start_token_id)

    model = _from_pretrained(AutoModelForSeq2SeqLM, model_id, dtype=torch.float32)
    loader = {
        "marian": MarianLoader, "mbart": converters.MBartLoader, "m2m_100": converters.M2M100Loader,
    }[model.config.model_type]()
    spec = loader(model, tokenizer)
    spec.validate()
    spec.optimize(quantization="float16")
    spec.save(str(output))
    return model.config.max_position_embeddings


def _ctranslate2_checkpoint(model_id, tokenizer, name) -> tuple[Path, int]:
    checkpoint = _ROOT / "models" / "live-ct2" / name
    try:
        source = json.loads((checkpoint / _CT2_SOURCE).read_bytes())
        if source["model"] == model_id and type(source["max_positions"]) is int and (checkpoint / "model.bin").is_file():
            return checkpoint, source["max_positions"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=checkpoint.parent, prefix=f".{name}-") as directory:
        stage = Path(directory) / name
        stage.mkdir()
        max_positions = _convert_ctranslate2(model_id, tokenizer, stage)
        (stage / _CT2_SOURCE).write_text(
            json.dumps({"model": model_id, "max_positions": max_positions}), encoding="utf-8",
        )
        shutil.rmtree(checkpoint, ignore_errors=True)
        stage.rename(checkpoint)
    return checkpoint, max_positions


def _load_ctranslate2(model_id, tokenizer, name, use_cuda, input_limit, generation_options):
    try:
        import ctranslate2
    except ImportError:
        return None
    if use_cuda and not _load_cuda_blas():
        return None
    try:
        checkpoint, max_length = _ctranslate2_checkpoint(model_id, tokenizer, name)
        model = ctranslate2.Translator(str(checkpoint), device="cuda" if use_cuda else "cpu", intra_threads=4)
    except Exception:
        _LOG.warning("Could not initialize CTranslate2 for %s; using Transformers", model_id, exc_info=True)
        return None
    if input_limit is not None:
        max_length = min(max_length, input_limit)
    # Transformers' decoder start or forced first token both become CTranslate2's target prefix.
    start = generation_options.get("decoder_start_token_id", generation_options.get("forced_bos_token_id"))
    prefix = None if start is None else [tokenizer.convert_ids_to_tokens(start)]

    def translate(texts: list[str]) -> list[str]:
        batch = [
            tokenizer.convert_ids_to_tokens(tokenizer(text, truncation=True, max_length=max_length)["input_ids"])
            for text in texts
        ]
        results = model.translate_batch(
            batch, target_prefix=None if prefix is None else [prefix] * len(batch), beam_size=1,
            max_decoding_length=_MAX_NEW_TOKENS, max_batch_size=_BATCH_SIZE,
            suppress_sequences=[[tokenizer.pad_token]],
        )
        return [
            tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0]), skip_special_tokens=True)
            for result in results
        ]

    return translate


def _load_batched_seq2seq(
    model_id, tokenizer, *, fp16_directory=None, tokenizer_id=None, input_limit=None, ct2_name=None,
    **generation_options,
):
    import torch

    torch.set_num_threads(4)
    use_cuda = torch.cuda.is_available()
    if ct2_name is not None:
        translate = _load_ctranslate2(model_id, tokenizer, ct2_name, use_cuda, input_limit, generation_options)
        if translate is not None:
            return translate

    from transformers import AutoModelForSeq2SeqLM

    device = "cuda:0" if use_cuda else "cpu"
    checkpoint = None
    if use_cuda and fp16_directory:
        checkpoint = _local_fp16_checkpoint(fp16_directory, model_id, tokenizer_id)
    if checkpoint is None:
        model = _from_pretrained(
            AutoModelForSeq2SeqLM, model_id, dtype=torch.float16 if use_cuda else torch.float32
        ).eval().to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, dtype=torch.float16, device_map=device,
            local_files_only=True, use_safetensors=True,
        ).eval()

    max_length = model.config.max_position_embeddings
    if input_limit is not None:
        max_length = min(max_length, input_limit)

    def translate(texts: list[str]) -> list[str]:
        translations = []
        with torch.inference_mode():
            for first in range(0, len(texts), _BATCH_SIZE):
                inputs = tokenizer(
                    texts[first:first + _BATCH_SIZE],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(device)
                outputs = model.generate(
                    **inputs,
                    **generation_options,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    num_beams=1,
                    do_sample=False,
                )
                translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return translations

    return translate


def _load_mbart_translator():
    from transformers import AutoTokenizer

    model_id = resolve_model(_ROOT, "live-translate-en-vi")
    tokenizer = _from_pretrained(AutoTokenizer, model_id, src_lang="en_XX")
    decoder_start = tokenizer.convert_tokens_to_ids("vi_VN")
    if not isinstance(decoder_start, int) or decoder_start == tokenizer.unk_token_id:
        raise ValueError("The selected mBART tokenizer must define the vi_VN target token")
    fp16_directory = _ROOT / "models" / "live-mbart-fp16"
    return _load_batched_seq2seq(
        model_id, tokenizer, fp16_directory=fp16_directory, tokenizer_id=model_id, ct2_name="vinai-en-vi",
        decoder_start_token_id=decoder_start,
    )


def _load_mitsua_translator():
    from transformers import AutoTokenizer

    model_id = resolve_model(_ROOT, JA_MODEL_KEYS[DEFAULT_JA_MODEL])
    tokenizer = _from_pretrained(AutoTokenizer, model_id)
    return _load_batched_seq2seq(model_id, tokenizer, ct2_name="mitsua-ja-en")


def _load_nllb_translator():
    from transformers import AutoTokenizer

    model_id = resolve_model(_ROOT, JA_MODEL_KEYS["nllb"])
    tokenizer = _from_pretrained(AutoTokenizer, model_id, src_lang="jpn_Jpan")
    english = tokenizer.convert_tokens_to_ids("eng_Latn")
    if not isinstance(english, int) or english == tokenizer.unk_token_id:
        raise ValueError("The selected NLLB tokenizer must define the eng_Latn target token")
    return _load_batched_seq2seq(
        model_id, tokenizer, input_limit=512, ct2_name="nllb-ja-en", forced_bos_token_id=english,
    )


def _load_custom_nllb_translator():
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    checkpoint = Path(resolve_model(_ROOT, NLLB_CUSTOM_MODEL_KEY))
    if not checkpoint.is_dir():
        raise RuntimeError("Install NLLB Custom ONNX from Setup before using this translation flow.")
    config = json.loads((checkpoint / "config.json").read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang="jpn_Jpan", local_files_only=True)
    vietnamese = tokenizer.convert_tokens_to_ids("vie_Latn")
    if vietnamese == tokenizer.unk_token_id or not isinstance(vietnamese, int):
        raise ValueError("The NLLB Custom tokenizer must define the vie_Latn target token")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    encoder, decoder, cached_decoder = (
        ort.InferenceSession(str(checkpoint / name), sess_options=options, providers=["CPUExecutionProvider"])
        for name in NLLB_ONNX_FILES
    )
    decoder_inputs = {session: [item.name for item in session.get_inputs()] for session in (decoder, cached_decoder)}
    decoder_outputs = {
        session: [item.name for item in session.get_outputs()] for session in (decoder, cached_decoder)
    }
    end = config["eos_token_id"]

    def translate(texts: list[str]) -> list[str]:
        translations = []
        for first in range(0, len(texts), _BATCH_SIZE):
            inputs = tokenizer(
                texts[first:first + _BATCH_SIZE], return_tensors="np", padding=True, truncation=True, max_length=256,
            )
            hidden, = encoder.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})
            size = len(inputs["input_ids"])
            feed = {"encoder_hidden_states": hidden, "encoder_attention_mask": inputs["attention_mask"]}
            token = np.full(size, config["decoder_start_token_id"], dtype=np.int64)
            finished = np.zeros(size, dtype=bool)
            generated = []
            for step in range(_MAX_NEW_TOKENS):
                session = cached_decoder if step else decoder
                feed["input_ids"] = token[:, None]
                outputs = dict(zip(
                    decoder_outputs[session], session.run(None, {key: feed[key] for key in decoder_inputs[session]}),
                    strict=True,
                ))
                # Cached steps update the self-attention keys/values; retain the encoder cache.
                feed.update({
                    name.replace("present.", "past_key_values.", 1): value
                    for name, value in outputs.items() if name.startswith("present.")
                })
                if step == 0:
                    token = np.full(size, vietnamese, dtype=np.int64)
                else:
                    token = np.where(finished, end, outputs["logits"][:, -1, :].argmax(axis=-1))
                generated.append(token)
                finished |= token == end
                if finished.all():
                    break
            translations.extend(tokenizer.batch_decode(np.stack(generated, axis=1), skip_special_tokens=True))
        return translations

    return translate


def _translate_batch(translator, texts):
    translated = translator(texts)
    if (
        not isinstance(translated, list)
        or len(translated) != len(texts)
        or any(not isinstance(line, str) for line in translated)
    ):
        raise ValueError("The live translator must return one string per input")
    return translated


def load_translator(ja_model=DEFAULT_JA_MODEL, *, direct_japanese=False):
    japanese_loader = _load_custom_nllb_translator if direct_japanese else {
        DEFAULT_JA_MODEL: _load_mitsua_translator, "nllb": _load_nllb_translator,
    }[ja_model]
    from_japanese = None
    en_to_vi = None

    def translate(texts: list[str]) -> list[str]:
        nonlocal from_japanese, en_to_vi
        translations = [""] * len(texts)
        english = {}
        japanese = {}
        for index, text in enumerate(texts):
            text = normalize("NFKC", " ".join(text.split()))
            if _JAPANESE_SCRIPT.search(text):
                japanese[index] = text
            elif _LATIN_SCRIPT.search(text):
                english[index] = text

        if japanese:
            if from_japanese is None:
                from_japanese = japanese_loader()
            lines = _translate_batch(from_japanese, list(japanese.values()))
            if direct_japanese:
                for index, line in zip(japanese, lines, strict=True):
                    translations[index] = line
            else:
                english.update(zip(japanese, lines, strict=True))

        if english:
            if en_to_vi is None:
                en_to_vi = _load_mbart_translator()
            positions = sorted(english)
            vietnamese = _translate_batch(en_to_vi, [english[index] for index in positions])
            for index, line in zip(positions, vietnamese, strict=True):
                translations[index] = line
        return translations

    return translate
