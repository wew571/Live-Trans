import json
import queue

from frontend.web_settings_host import WebSettingsWindow


class Root:
    def __init__(self):
        self.tasks = []
    def after(self, delay, callback):
        self.tasks.append(callback)
        return callback
    def after_cancel(self, callback):
        self.tasks.remove(callback)
    def tick(self):
        self.tasks.pop(0)()


class Pipe:
    def __init__(self):
        self.lines = []
    def write(self, text):
        self.lines.append(json.loads(text))
    def flush(self):
        pass
    def close(self):
        pass


class Process:
    def __init__(self):
        self.stdin = Pipe()
        self.returncode = None
    def poll(self):
        return self.returncode
    def terminate(self):
        self.returncode = 1
    def wait(self, timeout=None):
        return self.returncode


def make_host(monkeypatch):
    root, process, requests, events = Root(), Process(), [], []
    monkeypatch.setattr('frontend.web_settings_host.subprocess.Popen', lambda *a, **k: process)
    monkeypatch.setattr('frontend.web_settings_host.threading.Thread.start', lambda self: None)
    host = WebSettingsWindow(root, lambda *args: requests.append(args), lambda: events.append('fallback'), lambda: events.append('closed'))
    return root, process, requests, events, host


def test_bootstrap_and_async_response_are_dispatched_on_tick(monkeypatch):
    root, process, requests, events, host = make_host(monkeypatch)
    host._messages.put({'id': 3, 'method': 'bootstrap', 'draft': None})
    root.tick()
    assert requests[0][:2] == ('bootstrap', None)
    requests[0][2]({'ok': True})
    assert process.stdin.lines == [{'id': 3, 'result': {'ok': True}}]
    assert events == []


def test_exit_before_bootstrap_falls_back_but_explicit_destroy_does_not(monkeypatch):
    root, process, requests, events, host = make_host(monkeypatch)
    process.returncode = 1
    host._reader_done.set()
    root.tick()
    assert events == ['fallback', 'closed']
    root, process, requests, events, host = make_host(monkeypatch)
    host.destroy()
    assert events == ['closed']
    assert not host.winfo_exists()
    assert root.tasks == []


def test_focus_targets_existing_child_and_unknown_requests_are_rejected(monkeypatch):
    root, process, requests, events, host = make_host(monkeypatch)
    host.focus_set()
    host._messages.put({'id': 4, 'method': 'arbitrary', 'draft': {}})
    root.tick()
    assert process.stdin.lines[0] == {'command': 'focus'}
    assert requests == []
    assert process.stdin.lines[1]['id'] == 4
    assert 'error' in process.stdin.lines[1]


def test_child_rejects_bootstrap_when_renderer_fell_back(monkeypatch):
    import sys
    from types import SimpleNamespace
    from frontend.web_settings_host import _ChildBridge
    import pytest
    monkeypatch.setitem(sys.modules, 'webview', SimpleNamespace(renderer='mshtml'))
    with pytest.raises(RuntimeError):
        _ChildBridge(Pipe()).bootstrap()


def test_manage_rules_keeps_web_settings_visible_and_passes_its_window_handle():
    from unittest.mock import Mock
    from frontend.web_settings_host import _ChildBridge

    bridge = _ChildBridge(Pipe())
    bridge._window = Mock()
    bridge._window.native.Handle.ToInt64.return_value = 123
    bridge._request = Mock(return_value={"rule_name": "Qwen"})

    assert bridge.manage_rules("Extreme") == {"rule_name": "Qwen"}
    bridge._window.hide.assert_not_called()
    bridge._request.assert_called_once_with("manage_rules", {"settings_hwnd": 123, "theme_preset": "Extreme"})
    assert not bridge._rules_open.is_set()


def test_malformed_method_is_rejected_without_breaking_poll(monkeypatch):
    root, process, requests, events, host = make_host(monkeypatch)
    host._messages.put({'id': 8, 'method': [], 'draft': {}})
    root.tick()
    assert process.stdin.lines == [{'id': 8, 'error': 'Invalid request'}]
    assert host.winfo_exists()


def test_initial_validation_completes_readiness_and_cooperative_close_skips_fallback(monkeypatch):
    root, process, requests, events, host = make_host(monkeypatch)
    host._messages.put({'id': 1, 'method': 'bootstrap', 'draft': None})
    root.tick()
    assert not host._ready
    host._messages.put({'id': 2, 'method': 'validate', 'draft': {}})
    root.tick()
    assert host._ready
    root, process, requests, events, host = make_host(monkeypatch)
    host._messages.put({'event': 'closing'})
    process.returncode = 0
    host._reader_done.set()
    root.tick()
    assert events == ['closed']

