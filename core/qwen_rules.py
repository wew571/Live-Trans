"""Local storage for named Qwen translation rules."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from time import time_ns
from uuid import uuid4


_SLOT_ID = re.compile(r"[a-f0-9]{32}")


@dataclass(frozen=True)
class QwenRule:
    id: str
    name: str
    text: str

    def __post_init__(self):
        if not isinstance(self.id, str) or not _SLOT_ID.fullmatch(self.id):
            raise ValueError("Invalid rule slot ID.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Enter a name for the rule slot.")
        if not isinstance(self.text, str):
            raise ValueError("Rule content must be text.")

    @classmethod
    def from_json(cls, value) -> "QwenRule | None":
        if isinstance(value, dict):
            try:
                return cls(value.get("id"), value.get("name"), value.get("text"))
            except ValueError:
                pass
        return None


def _slot_directory(root: Path) -> Path:
    return root / "models" / "live-ocr" / "qwen-rules"


def load_rule_slots(root: Path) -> list[QwenRule]:
    slots = []
    for path in sorted(_slot_directory(root).glob("*.json")):
        try:
            slot = QwenRule.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if slot is not None and slot.id == path.stem:
            slots.append(slot)
    return slots


def save_rule_slot(root: Path, *, name: str, text: str, slot_id: str | None = None) -> QwenRule:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Enter a name for the rule slot.")
    directory = _slot_directory(root)
    slot_id_value = slot_id
    if slot_id_value is None:
        # Windows can return the same clock tick for two writes.
        last = max(
            (int(path.stem[:16], 16) for path in directory.glob("*.json") if _SLOT_ID.fullmatch(path.stem)),
            default=0,
        )
        slot_id_value = f"{max(time_ns(), last + 1):016x}{uuid4().hex[16:]}"
    slot = QwenRule(slot_id_value, name.strip(), text)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slot.id}.json"
    if slot_id is not None and not path.is_file():
        raise ValueError("This rule slot is missing.")
    temporary = None
    try:
        with NamedTemporaryFile(dir=directory, encoding="utf-8", mode="w", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(asdict(slot), stream, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return slot
