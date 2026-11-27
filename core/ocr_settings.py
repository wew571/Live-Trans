"""Persist Live Trans model selections in the existing settings file."""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from core.qwen_rules import QwenRule

OCR_SIZES = ("small", "medium")
DEFAULT_JA_MODEL = "mitsua"
FLOW_PIVOT = "pivot"
FLOW_DIRECT = "direct"
FLOW_NLLB_VINAI = "nllb-vinai"
_FLOWS = (FLOW_PIVOT, FLOW_DIRECT, FLOW_NLLB_VINAI)
INTERFACE_MINIMAL = "minimal"
INTERFACE_ADVANCED = "advanced"
_SETTINGS_INTERFACES = (INTERFACE_MINIMAL, INTERFACE_ADVANCED)
QWEN_2B = "2b"
QWEN_4B = "4b"
QWEN_9B = "9b"
QWEN_27B = "27b"
QWEN_MODEL_LABELS = {
    QWEN_2B: "Qwen3.5-2B Q4_K_M",
    QWEN_4B: "Qwen3.5-4B Q4_K_M",
    QWEN_9B: "Qwen3.5-9B Q4_K_M",
    QWEN_27B: "Qwen3.8-27B Q4_K_M",
}
QWEN_CONTEXT_MIN = 1000
QWEN_CONTEXT_MAX = 25000
QWEN_CONTEXT_DEFAULT = 3000
QWEN_BACKEND_VULKAN = "vulkan"
QWEN_BACKEND_CUDA_PYTHON = "cuda-python"
QWEN_BACKEND_LABELS = {
    QWEN_BACKEND_VULKAN: "llama.cpp Vulkan",
    QWEN_BACKEND_CUDA_PYTHON: "llama-cpp-python CUDA",
}
CROP_COUNTS = (1, 2, 3)
HISTORY_TURNS_MAX = 50
POST_TRANSLATION_DELAY_MAX = 5000
DEFAULT_TRANSLATION_COLOR = "#f8fafc"
_DEFAULT_CROP_RULE_SLOTS = ("",) * len(CROP_COUNTS)
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
    animation_enabled: bool = False
    settings_interface: str = INTERFACE_MINIMAL
    crop_count: int = 1
    history_turns: int = 2
    crop_rule_slots: tuple[str, ...] = _DEFAULT_CROP_RULE_SLOTS
    qwen_backend: str = QWEN_BACKEND_VULKAN
    translation_color: str | None = None
    post_translation_delay_ms: int = 0

    def __post_init__(self):
        if self.det not in OCR_SIZES or self.rec not in OCR_SIZES:
            raise ValueError("PP-OCRv6 model size must be small or medium.")
        if not isinstance(self.ja_model, str) or self.ja_model not in JA_MODEL_KEYS:
            raise ValueError("Unknown Japanese translation model.")
        if self.flow not in _FLOWS:
            raise ValueError("Unknown live translation flow.")
        if not _valid_context(self.qwen_context):
            raise ValueError("Qwen context must be an integer from 1000 to 25000 tokens.")
        if self.qwen_rule is not None and not isinstance(self.qwen_rule, QwenRule):
            raise ValueError("Invalid Qwen rule.")
        if not isinstance(self.qwen_model, str) or self.qwen_model not in QWEN_MODEL_LABELS:
            raise ValueError("Unknown Qwen translation model.")
        if not isinstance(self.animation_enabled, bool):
            raise ValueError("Animation must be enabled or disabled.")
        if self.settings_interface not in _SETTINGS_INTERFACES:
            raise ValueError("Unknown settings interface.")
        if type(self.crop_count) is not int or self.crop_count not in CROP_COUNTS:
            raise ValueError("Choose one to three crop regions.")
        if not valid_history_turns(self.history_turns):
            raise ValueError(f"Enter a whole number from 0 to {HISTORY_TURNS_MAX} for dialogue turns.")
        if self.crop_rule_slots != _crop_values(self.crop_rule_slots, _DEFAULT_CROP_RULE_SLOTS):
            raise ValueError("Invalid crop rule slots.")
        if not isinstance(self.qwen_backend, str) or self.qwen_backend not in QWEN_BACKEND_LABELS:
            raise ValueError("Unknown Qwen backend.")
        if not valid_post_translation_delay(self.post_translation_delay_ms):
            raise ValueError("Post-translation delay must be an integer from 0 to 5000 ms.")
        if self.translation_color is not None and not _valid_translation_color(self.translation_color):
            raise ValueError("Translation color must use #RRGGBB.")

    def same_pipeline_as(self, other: "OCRSettings") -> bool:
        return replace(
            self, animation_enabled=other.animation_enabled, settings_interface=other.settings_interface,
            translation_color=other.translation_color, post_translation_delay_ms=other.post_translation_delay_ms,
            crop_count=other.crop_count, history_turns=other.history_turns, qwen_context=other.qwen_context,
        ) == other

    @property
    def preset(self) -> str | None:
        model = self.qwen_model if self.flow == FLOW_DIRECT else self.ja_model
        for name, preset in PRESETS.items():
            if (self.det, self.rec, self.flow) != (preset.det, preset.rec, preset.flow):
                continue
            preset_model = preset.qwen_model if preset.flow == FLOW_DIRECT else preset.ja_model
            if self.flow == FLOW_NLLB_VINAI or model == preset_model:
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
                if flow not in _FLOWS:
                    flow = FLOW_PIVOT
                context = values.get("qwen_context")
                if not _valid_context(context):
                    context = QWEN_CONTEXT_DEFAULT
                rule = QwenRule.from_json(values.get("qwen_rule"))
                qwen_model = values.get("qwen_model")
                if not isinstance(qwen_model, str) or qwen_model not in QWEN_MODEL_LABELS:
                    qwen_model = QWEN_4B
                interface = values.get("settings_interface")
                crop_count = values.get("crop_count")
                history_turns = values.get("history_turns")
                backend = values.get("qwen_backend")
                color = values.get("translation_color")
                delay = values.get("post_translation_delay_ms")
                if not isinstance(backend, str) or backend not in QWEN_BACKEND_LABELS:
                    backend = QWEN_BACKEND_VULKAN
                return cls(
                    values.get("det", "small"), values.get("rec", "small"),
                    ja_model, flow, context, rule, qwen_model,
                    animation_enabled=values.get("animation_enabled") is True,
                    settings_interface=interface if interface in _SETTINGS_INTERFACES else INTERFACE_MINIMAL,
                    crop_count=crop_count if type(crop_count) is int and crop_count in CROP_COUNTS else 1,
                    history_turns=history_turns if valid_history_turns(history_turns) else 2,
                    crop_rule_slots=_crop_values(values.get("crop_rule_slots"), _DEFAULT_CROP_RULE_SLOTS),
                    qwen_backend=backend,
                    translation_color=color if _valid_translation_color(color) else None,
                    post_translation_delay_ms=delay if valid_post_translation_delay(delay) else 0,
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


def valid_post_translation_delay(value) -> bool:
    return type(value) is int and 0 <= value <= POST_TRANSLATION_DELAY_MAX


def valid_history_turns(value) -> bool:
    return type(value) is int and 0 <= value <= HISTORY_TURNS_MAX


def _valid_translation_color(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", value) is not None


def _crop_values(values, defaults):
    if not isinstance(values, (list, tuple)) or len(values) != len(defaults):
        return defaults
    return tuple(
        value if isinstance(value, str) else default
        for value, default in zip(values, defaults, strict=True)
    )


PRESETS = {
    "Flash": OCRSettings(),
    "Medium": OCRSettings(det="medium", rec="small", flow=FLOW_NLLB_VINAI),
    "High": OCRSettings("medium", "medium", flow=FLOW_DIRECT, qwen_model=QWEN_2B),
    "Ultra": OCRSettings("medium", "medium", flow=FLOW_DIRECT),
    "Extreme": OCRSettings("medium", "medium", flow=FLOW_DIRECT, qwen_model=QWEN_9B),
    "Ultimate": OCRSettings("medium", "medium", flow=FLOW_DIRECT, qwen_model=QWEN_27B),
}
