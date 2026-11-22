from pathlib import Path
from unittest.mock import Mock

import pytest


def test_qwen_sends_text_without_thinking_and_preserves_positions(monkeypatch):
    from core.live_qwen import QwenTranslator

    translator = QwenTranslator(
        Path("test.gguf"), Path("llama-server.exe"), 3000, rules="Giữ tên nhân vật.\nXưng tôi.",
    )
    start = Mock()
    request = Mock(side_effect=[
        {"choices": [{"message": {"content": "Cài đặt"}}]},
        {"choices": [{"message": {"content": "Tiếp tục"}}]},
    ])
    monkeypatch.setattr(translator, "_start", start)
    monkeypatch.setattr(translator, "_request", request)
    assert translator(["設定\nです", "", "Continue", "한글"]) == ["Cài đặt", "", "Tiếp tục", ""]
    start.assert_called_once()
    for call, source in zip(request.call_args_list, ["設定\nです", "Continue"], strict=True):
        endpoint, payload = call.args
        assert endpoint == "/v1/chat/completions"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        system, dialogue = payload["messages"]
        assert "Giữ tên nhân vật.\nXưng tôi." in system["content"]
        assert dialogue["content"].endswith(source)


def test_qwen_uses_only_the_base_prompt_when_no_rule_is_selected(monkeypatch):
    from core.live_qwen import QwenTranslator

    translator = QwenTranslator(Path("test.gguf"), Path("llama-server.exe"), 3000)
    request = Mock(return_value={"choices": [{"message": {"content": "Tiếp tục"}}]})
    monkeypatch.setattr(translator, "_start", Mock())
    monkeypatch.setattr(translator, "_request", request)
    translator(["Continue"])
    prompt = request.call_args.args[1]["messages"][0]["content"]
    assert prompt.splitlines() == [
        "Dịch lời thoại tiếng Nhật hoặc tiếng Anh sang phụ đề tiếng Việt tự nhiên, sát ý.",
        "Đầu vào là tên người nói và các dòng OCR thuộc một lượt thoại. Đọc các dòng như một đoạn liên tục.",
        "Chỉ trả về bản dịch của lời thoại, bỏ nhãn tên người nói. Giữ sắc thái, ngập ngừng và dấu ngoặc độc thoại nếu có.",
        "Không tự thêm chủ ngữ, sự kiện hay lời giải thích khi bản gốc mơ hồ.",
    ]


def test_qwen_rejects_thinking_in_the_response(monkeypatch):
    from core.live_qwen import QwenTranslator

    translator = QwenTranslator(Path("test.gguf"), Path("llama-server.exe"), 3000)
    monkeypatch.setattr(translator, "_start", Mock())
    monkeypatch.setattr(translator, "_request", Mock(return_value={
        "choices": [{"message": {"content": "Tin nhắn", "reasoning_content": "Thinking"}}],
    }))
    with pytest.raises(RuntimeError, match="thinking"):
        translator(["Message"])


def test_long_ocr_gets_more_output_tokens_and_marks_a_truncated_translation(monkeypatch):
    from core.live_qwen import QwenTranslator

    translator = QwenTranslator(Path("test.gguf"), Path("llama-server.exe"), 3000)
    request = Mock(return_value={
        "choices": [{"message": {"content": "Bản dịch còn thiếu"}, "finish_reason": "length"}],
    })
    monkeypatch.setattr(translator, "_start", Mock())
    monkeypatch.setattr(translator, "_request", request)
    assert translator(["Dialogue from OCR. " * 15]) == ["Bản dịch còn thiếu…"]
    assert request.call_args.args[1]["max_tokens"] > 132
