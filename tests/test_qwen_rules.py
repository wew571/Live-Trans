from dataclasses import replace
from uuid import UUID

import pytest


def test_named_rules_can_be_created_renamed_and_reopened(tmp_path, monkeypatch):
    from core import qwen_rules
    from core.qwen_rules import load_rule_slots, save_rule_slot

    monkeypatch.setattr(qwen_rules, "time_ns", lambda: 1234)
    ids = iter(("f" * 32, "0" * 32))
    monkeypatch.setattr(qwen_rules, "uuid4", lambda: UUID(next(ids)))
    assert load_rule_slots(tmp_path) == []
    first = save_rule_slot(tmp_path, name="First game", text="Keep character names.\nUse short sentences.")
    second = save_rule_slot(tmp_path, name="Second game", text="")
    renamed = save_rule_slot(tmp_path, slot_id=first.id, name="Renamed", text=first.text)
    assert renamed.id == first.id
    assert load_rule_slots(tmp_path) == [renamed, second]
    with pytest.raises(ValueError):
        save_rule_slot(tmp_path, name="  ", text="")
    with pytest.raises(ValueError):
        save_rule_slot(tmp_path, slot_id="../settings", name="Invalid ID", text="")


def test_applied_snapshot_stays_the_same_until_settings_are_saved(tmp_path):
    from core.ocr_settings import OCRSettings, PRESETS
    from core.qwen_rules import save_rule_slot
    from core.setup_models import available_model_settings

    assert OCRSettings.load(tmp_path).qwen_rule is None
    slot = save_rule_slot(tmp_path, name="One game", text="Use em.")
    applied = replace(PRESETS["XHigh"], qwen_rule=slot)
    applied.save(tmp_path)
    edited = save_rule_slot(tmp_path, slot_id=slot.id, name="One game", text="Use tôi.")
    assert OCRSettings.load(tmp_path) == applied
    assert available_model_settings(tmp_path).qwen_rule == slot
    replace(applied, qwen_rule=edited).save(tmp_path)
    assert OCRSettings.load(tmp_path).qwen_rule == edited
    replace(OCRSettings(det="medium"), qwen_rule=edited).save(tmp_path)
    assert available_model_settings(tmp_path).qwen_rule == edited
