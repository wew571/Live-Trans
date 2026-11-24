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
        {"choices": [{"message": {"content": "Cài đặt"}}], "usage": {"total_tokens": 199}},
        {"choices": [{"message": {"content": "Tiếp tục"}}], "usage": {"total_tokens": 2255}},
    ])
    monkeypatch.setattr(translator, "_start", start)
    monkeypatch.setattr(translator, "_request", request)
    assert translator(["設定\nです", "", "Continue", "한글"]) == ["Cài đặt", "", "Tiếp tục", ""]
    assert translator.last_token_count == 2255
    start.assert_called_once()
    for call, source in zip(request.call_args_list, ["設定\nです", "Continue"], strict=True):
        endpoint, payload = call.args
        assert endpoint == "/v1/chat/completions"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        system, dialogue = payload["messages"]
        assert "Giữ tên nhân vật.\nXưng tôi." in system["content"]
        assert dialogue["content"].endswith(source)
    assert translator(["한글"]) == [""]
    assert translator.last_token_count is None


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

def test_contextual_qwen_uses_recent_dialogue_and_current_crop_rules(monkeypatch):
    from core.live_qwen import QwenTranslator

    translator = QwenTranslator(
        Path("test.gguf"), Path("llama-server.exe"), 3000,
        rules="Role: Speaker\nUse official names.",
    )
    request = Mock(return_value={"choices": [{"message": {"content": "Xin vài bông nhé."}}]})
    monkeypatch.setattr(translator, "_start", Mock())
    monkeypatch.setattr(translator, "_request", request)

    assert translator.translate_contextual(
        "バラを少しだけ", role="dialogue", rules="Use a gentle tone.", speaker="Curren Bouquetd'or",
        history=(("Curren Bouquetd'or", "水やり", "Tưới nước."),),
    ) == "Xin vài bông nhé."
    messages = request.call_args.args[1]["messages"]
    assert "Use official names." in messages[0]["content"]
    assert "Role: Speaker" not in messages[0]["content"]
    assert any("水やり" in item["content"] for item in messages[:-1])
    assert any("Tưới nước." in item["content"] for item in messages[:-1])
    assert "Use a gentle tone." in messages[-1]["content"]
    assert "Curren Bouquetd'or" in messages[-1]["content"]
    assert "バラを少しだけ" in messages[-1]["content"]


def test_contextual_speaker_requests_a_name(monkeypatch):
    from core.live_qwen import QwenTranslator

    translator = QwenTranslator(Path("test.gguf"), Path("llama-server.exe"), 3000)
    request = Mock(return_value={"choices": [{"message": {"content": "Curren Bouquetd'or"}}]})
    monkeypatch.setattr(translator, "_start", Mock())
    monkeypatch.setattr(translator, "_request", request)

    assert translator.translate_contextual(
        "カレンブーケドール", role="speaker", rules="", speaker="", history=(),
    ) == "Curren Bouquetd'or"
    assert "chỉ trả về tên" in request.call_args.args[1]["messages"][-1]["content"].lower()


def test_contextual_qwen_retries_context_overflow_without_the_oldest_turn(monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError

    from core.live_qwen import QwenTranslator

    translator = QwenTranslator(Path("test.gguf"), Path("llama-server.exe"), 1000)
    request = Mock(side_effect=[
        HTTPError("http://localhost", 400, "Bad Request", {}, BytesIO(b"exceeds context size")),
        {"choices": [{"message": {"content": "Đây rồi."}}]},
    ])
    monkeypatch.setattr(translator, "_start", Mock())
    monkeypatch.setattr(translator, "_request", request)
    assert translator.translate_contextual(
        "Here it is.", role="dialogue", rules="", speaker="",
        history=(("", "oldest", "cũ nhất"), ("", "recent turn", "lượt gần nhất")),
    ) == "Đây rồi."
    retry = request.call_args.args[1]["messages"]
    assert len(retry) == 4
    assert "recent turn" in retry[1]["content"]


def test_python_cuda_uses_qwen_template_without_thinking(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from core import live_qwen_cuda

    model = Mock()
    model.n_ctx.return_value = 1000
    model.tokenize.return_value = [10, 20, 30]
    model.create_completion.return_value = {
        "choices": [{"text": "Tưới nước xong rồi.", "finish_reason": "stop"}],
        "usage": {"total_tokens": 222},
    }
    formatter = Mock(return_value=SimpleNamespace(prompt="formatted prompt", stop=["<|im_end|>"]))
    weights = tmp_path / "qwen.gguf"
    weights.touch()
    monkeypatch.setattr(live_qwen_cuda, "_open_cuda_model", Mock(return_value=(model, formatter)))
    translator = live_qwen_cuda.PythonCudaQwenTranslator(weights, tmp_path, 1000, rules="Giữ tên.")
    try:
        assert translator.translate_contextual(
            "水やり", role="dialogue", rules="Giọng nhẹ nhàng.", speaker="Curren", history=(),
        ) == "Tưới nước xong rồi."
        assert formatter.call_args.kwargs["enable_thinking"] is False
        assert "Giữ tên." in formatter.call_args.kwargs["messages"][0]["content"]
        assert "Giọng nhẹ nhàng." in formatter.call_args.kwargs["messages"][-1]["content"]
        assert model.create_completion.call_args.args == ([10, 20, 30],)
        assert translator.last_token_count == 222
    finally:
        translator.close()
    model.close.assert_called_once()
