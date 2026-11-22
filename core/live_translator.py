"""Translate English or Japanese OCR lines into Vietnamese."""

import json
from pathlib import Path
import re
from unicodedata import normalize

from core.ocr_settings import DEFAULT_JA_MODEL, JA_MODEL_KEYS
from core.setup_models import _file_present, _index_shards, resolve_model

_ROOT = Path(__file__).resolve().parents[1]
_BATCH_SIZE = 32
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


def _load_batched_seq2seq(
    model_id, tokenizer, *, fp16_directory=None, tokenizer_id=None, input_limit=None,
    **generation_options,
):
    import torch
    from transformers import AutoModelForSeq2SeqLM

    torch.set_num_threads(4)
    use_cuda = torch.cuda.is_available()
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
                    max_new_tokens=96,
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
        model_id, tokenizer, fp16_directory=fp16_directory, tokenizer_id=model_id,
        decoder_start_token_id=decoder_start,
    )


def _load_mitsua_translator():
    from transformers import AutoTokenizer

    model_id = resolve_model(_ROOT, JA_MODEL_KEYS[DEFAULT_JA_MODEL])
    tokenizer = _from_pretrained(AutoTokenizer, model_id)
    return _load_batched_seq2seq(model_id, tokenizer)


def _load_nllb_translator():
    from transformers import AutoTokenizer

    model_id = resolve_model(_ROOT, JA_MODEL_KEYS["nllb"])
    tokenizer = _from_pretrained(AutoTokenizer, model_id, src_lang="jpn_Jpan")
    english = tokenizer.convert_tokens_to_ids("eng_Latn")
    if not isinstance(english, int) or english == tokenizer.unk_token_id:
        raise ValueError("The selected NLLB tokenizer must define the eng_Latn target token")
    return _load_batched_seq2seq(model_id, tokenizer, input_limit=512, forced_bos_token_id=english)


def _translate_batch(translator, texts):
    translated = translator(texts)
    if (
        not isinstance(translated, list)
        or len(translated) != len(texts)
        or any(not isinstance(line, str) for line in translated)
    ):
        raise ValueError("The live translator must return one string per input")
    return translated


def load_translator(ja_model=DEFAULT_JA_MODEL):
    japanese_loader = {DEFAULT_JA_MODEL: _load_mitsua_translator, "nllb": _load_nllb_translator}[ja_model]
    ja_to_en = None
    en_to_vi = None

    def translate(texts: list[str]) -> list[str]:
        nonlocal ja_to_en, en_to_vi
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
            if ja_to_en is None:
                ja_to_en = japanese_loader()
            pivot = _translate_batch(ja_to_en, list(japanese.values()))
            english.update(zip(japanese, pivot, strict=True))

        if english:
            if en_to_vi is None:
                en_to_vi = _load_mbart_translator()
            positions = sorted(english)
            vietnamese = _translate_batch(en_to_vi, [english[index] for index in positions])
            for index, line in zip(positions, vietnamese, strict=True):
                translations[index] = line
        return translations

    return translate
