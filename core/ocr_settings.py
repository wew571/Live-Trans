"""Persist Live Trans model selections in the existing settings file."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from core.qwen_rules import QwenRule

OCR_SIZES = ("small", "medium")
DEFAULT_JA_MODEL = "mitsua"
FLOW_PIVOT = "pivot"
FLOW_DIRECT = "direct"
QWEN_2B = "2b"
QWEN_4B = "4b"
QWEN_MODEL_LABELS = {
    QWEN_2B: "Qwen3.5-2B Q4_K_M",
    QWEN_4B: "Qwen3.5-4B Q4_K_M",
}
QWEN_CONTEXT_MIN = 1000
QWEN_CONTEXT_MAX = 25000
QWEN_CONTEXT_DEFAULT = 3000
JA_MODEL_KEYS = {
    DEFAULT_JA_MODEL: "translate-ja-en",
    "nllb": "translate-ja-en-nllb",
}


def ocr_model_key(role: str, size: str) -> str:
    return f"live-ocr-{role}" if size == "small" else f"live-ocr-{size}-{role}"


@dataclass(frozen=True)
class OCRSettings:
    det: str = "small"
    rec: str = "small"
    ja_model: str = DEFAULT_JA_MODEL
    flow: str = FLOW_PIVOT
    qwen_context: int = QWEN_CONTEXT_DEFAULT
    qwen_rule: QwenRule | None = None
    qwen_model: str = QWEN_4B

    def __post_init__(self):
        if self.det not in OCR_SIZES or self.rec not in OCR_SIZES:
            raise ValueError("PP-OCRv6 model size must be small or medium.")
        if not isinstance(self.ja_model, str) or self.ja_model not in JA_MODEL_KEYS:
            raise ValueError("Unknown Japanese translation model.")
        if self.flow not in (FLOW_PIVOT, FLOW_DIRECT):
            raise ValueError("Unknown live translation flow.")
        if not _valid_context(self.qwen_context):
            raise ValueError("Qwen context must be an integer from 1000 to 25000 tokens.")
        if self.qwen_rule is not None and not isinstance(self.qwen_rule, QwenRule):
            raise ValueError("Invalid Qwen rule.")
        if not isinstance(self.qwen_model, str) or self.qwen_model not in QWEN_MODEL_LABELS:
            raise ValueError("Unknown Qwen translation model.")

    @property
    def preset(self) -> str | None:
        model = self.qwen_model if self.flow == FLOW_DIRECT else self.ja_model
        for name, preset in PRESETS.items():
            if (self.det, self.rec, self.flow) != (preset.det, preset.rec, preset.flow):
                continue
            preset_model = preset.qwen_model if preset.flow == FLOW_DIRECT else preset.ja_model
            if model == preset_model:
                return name
        return None

    @staticmethod
    def _path(root: Path) -> Path:
        return root / "models" / "live-ocr" / "settings.json"

    @classmethod
    def load(cls, root: Path) -> "OCRSettings":
        try:
            values = json.loads(cls._path(root).read_text(encoding="utf-8"))
            if isinstance(values, dict):
                ja_model = values.get("ja_model")
                if not isinstance(ja_model, str) or ja_model not in JA_MODEL_KEYS:
                    ja_model = DEFAULT_JA_MODEL
                flow = values.get("flow")
                if flow not in (FLOW_PIVOT, FLOW_DIRECT):
                    flow = FLOW_PIVOT
                context = values.get("qwen_context")
                if not _valid_context(context):
                    context = QWEN_CONTEXT_DEFAULT
                rule = QwenRule.from_json(values.get("qwen_rule"))
                qwen_model = values.get("qwen_model")
                if not isinstance(qwen_model, str) or qwen_model not in QWEN_MODEL_LABELS:
                    qwen_model = QWEN_4B
                return cls(
                    values.get("det", "small"), values.get("rec", "small"),
                    ja_model, flow, context, rule, qwen_model,
                )
        except (OSError, ValueError, TypeError):
            pass
        return cls()

    def save(self, root: Path) -> None:
        path = self._path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, encoding="utf-8", mode="w", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(asdict(self), stream)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _valid_context(context) -> bool:
    return type(context) is int and QWEN_CONTEXT_MIN <= context <= QWEN_CONTEXT_MAX


PRESETS = {
    "Flash": OCRSettings(),
    "Medium": OCRSettings(det="medium", rec="small", ja_model="nllb"),
    "High": OCRSettings("medium", "medium", flow=FLOW_DIRECT, qwen_model=QWEN_2B),
    "XHigh": OCRSettings("medium", "medium", flow=FLOW_DIRECT),
}
