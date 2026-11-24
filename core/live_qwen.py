"""Text-only Qwen inference in a local llama.cpp server owned by the live worker."""

from __future__ import annotations

from contextlib import suppress
import json
import os
from pathlib import Path
import re
from secrets import token_urlsafe
import socket
import subprocess
from tempfile import TemporaryFile
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from core.qwen_rules import parse_crop_rule


_SOURCE_SCRIPT = re.compile(r"[A-Za-z\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]")
_SYSTEM_PROMPT = """Dịch lời thoại tiếng Nhật hoặc tiếng Anh sang phụ đề tiếng Việt tự nhiên, sát ý.
Đầu vào là tên người nói và các dòng OCR thuộc một lượt thoại. Đọc các dòng như một đoạn liên tục.
Chỉ trả về bản dịch của lời thoại, bỏ nhãn tên người nói. Giữ sắc thái, ngập ngừng và dấu ngoặc độc thoại nếu có.
Không tự thêm chủ ngữ, sự kiện hay lời giải thích khi bản gốc mơ hồ."""


class QwenContextLimitError(RuntimeError):
    """The prompt and reserved output exceed the configured context."""


def _kill_with_worker(process):
    # Closing the worker's handle also ends llama-server if the worker is killed
    # while handling a request or loading the model.
    import win32api
    import win32con
    import win32job

    job = win32job.CreateJobObject(None, "")
    try:
        limits = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
        limits["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, limits)
        handle = win32api.OpenProcess(win32con.PROCESS_SET_QUOTA | win32con.PROCESS_TERMINATE, False, process.pid)
        try:
            win32job.AssignProcessToJobObject(job, handle)
        finally:
            handle.Close()
    except BaseException:
        job.Close()
        raise
    return job


class QwenTranslator:
    """Keep one Qwen model loaded for a live session."""

    def __init__(self, model_path: Path, server_path: Path | None, context_size: int, *, rules: str = ""):
        self._model_path = model_path.resolve()
        self._server_path = server_path.resolve() if server_path is not None else None
        self._context_size = context_size
        if not isinstance(rules, str):
            raise ValueError("Qwen rules must be text.")
        _, rules = parse_crop_rule(rules)
        self._system_prompt = _SYSTEM_PROMPT
        if rules.strip():
            self._system_prompt += (
                "\n\nÁp dụng các quy tắc dịch tùy chỉnh sau khi phù hợp với chỉ dẫn dịch ở trên.\n"
                + rules.strip()
            )
        self._opener = build_opener(ProxyHandler({}))
        self._api_key = token_urlsafe(32)
        self._base_url = ""
        self._process = None
        self._job = None
        self._log = None
        self.last_token_count: int | None = None

    def __call__(self, texts: list[str]) -> list[str]:
        self.last_token_count = None
        translations = [""] * len(texts)
        inputs = [(index, text.strip()) for index, text in enumerate(texts) if _SOURCE_SCRIPT.search(text)]
        if inputs:
            self._start()
        for index, text in inputs:
            translations[index] = self._translate(text)
        return translations

    def translate_contextual(self, text: str, *, role: str, rules: str, speaker: str, history) -> str:
        self.last_token_count = None
        if not _SOURCE_SCRIPT.search(text):
            return ""
        self._start()
        system = (
            self._system_prompt
            + "\nKhi vùng hiện tại là nhãn người nói, chỉ chuẩn hóa tên riêng, giữ nguyên tên Latin "
              "chính thức nếu biết. Các lượt trước chỉ cung cấp ngữ cảnh. "
              "Quy tắc của vùng hiện tại được ưu tiên nếu khác quy tắc dịch chung."
        )
        current = (
            f"Vùng tên người nói. Chỉ trả về tên.\nTên OCR:\n{text}"
            if role == "speaker"
            else f"Người nói: {speaker or '(không rõ)'}\nLời thoại OCR:\n{text}"
        )
        if rules.strip():
            current = f"Quy tắc của vùng hiện tại:\n{rules.strip()}\n\n{current}"
        prefix = [{"role": "system", "content": system}]
        for old_speaker, source, translation in history:
            prefix.extend((
                {"role": "user", "content": f"Người nói: {old_speaker or '(không rõ)'}\nLời thoại OCR:\n{source}"},
                {"role": "assistant", "content": translation},
            ))
        while True:
            try:
                return self._translate(text, [*prefix, {"role": "user", "content": current}])
            except QwenContextLimitError:
                if len(prefix) <= 1:
                    raise
                del prefix[1:3]
            except RuntimeError as error:
                cause = error.__cause__
                if len(prefix) <= 1 or not isinstance(cause, HTTPError) or cause.code not in (400, 413):
                    raise
                with cause:
                    detail = cause.read(8192).lower()
                if b"context" not in detail:
                    raise
                del prefix[1:3]

    def _translate(self, text: str, messages=None) -> str:
        try:
            response = self._complete(
                messages if messages is not None else [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": f"Người nói: (không rõ)\nLời thoại OCR:\n{text}"},
                ],
                min(512, max(132, len(text) * 2)),
            )
            choice = response["choices"][0]
            message = choice["message"]
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError("Invalid content")
        except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
            raise RuntimeError("Qwen did not return a valid translation.") from error
        if message.get("reasoning_content") or re.search(
            r"</?think\s*>|^\s*Thinking Process\s*:", content, re.IGNORECASE,
        ):
            raise RuntimeError("Qwen returned thinking content with thinking disabled.")
        usage = response.get("usage")
        count = usage.get("total_tokens") if isinstance(usage, dict) else None
        self.last_token_count = count if type(count) is int and count >= 0 else None
        content = content.strip()
        if content and choice.get("finish_reason") == "length":
            return content.rstrip(".…") + "…"
        return content

    def _complete(self, messages, max_tokens):
        return self._request("/v1/chat/completions", {
            "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        })

    def _request(self, endpoint, payload=None, *, timeout=90):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        request = Request(
            f"{self._base_url}{endpoint}", data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with self._opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        if self._server_path is None or not self._server_path.is_file() or not self._model_path.is_file():
            raise RuntimeError("Qwen model or llama.cpp is missing. Install them in Setup.")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self._base_url = f"http://127.0.0.1:{port}"
        environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("LLAMA_ARG_")}
        self._log = TemporaryFile(mode="w+b")
        try:
            self._process = subprocess.Popen(
                [
                    str(self._server_path), "--model", str(self._model_path),
                    "--host", "127.0.0.1", "--port", str(port), "--api-key", self._api_key,
                    "--ctx-size", str(self._context_size), "--parallel", "1",
                    "--reasoning", "off", "--no-mmproj", "--fit", "on", "--fit-target", "384",
                    "-lv", "2",
                ],
                stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                cwd=self._server_path.parent, env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if os.name == "nt":
                self._job = _kill_with_worker(self._process)
            self._wait_until_ready()
        except BaseException:
            self.close()
            raise

    def _wait_until_ready(self):
        deadline = monotonic() + 70
        while self._process.poll() is None and monotonic() < deadline:
            try:
                if self._request("/health", timeout=2).get("status") == "ok":
                    return
            except (OSError, ValueError):
                pass
            sleep(0.2)
        self._log.seek(max(0, self._log.seek(0, os.SEEK_END) - 600))
        detail = self._log.read().decode("utf-8", errors="replace").strip().replace(self._api_key, "[key]")
        raise RuntimeError(f"Could not start Qwen. Check the runtime and available GPU memory. {detail}")

    def close(self):
        job, self._job = self._job, None
        process, self._process = self._process, None
        log, self._log = self._log, None
        try:
            if job is not None:
                job.Close()
            if process is not None:
                if process.poll() is None:
                    with suppress(ProcessLookupError):
                        process.kill()
                process.wait(timeout=5)
        finally:
            if log is not None:
                log.close()
