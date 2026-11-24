"""Isolated pywebview settings window and its private line-based bridge."""

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

_ROOT = Path(__file__).resolve().parent.parent
_STARTUP_TIMEOUT = 20.0
_METHODS = {'bootstrap', 'validate', 'save', 'manage_rules'}


class WebSettingsWindow:
    """Schedule child requests on Tk and allow asynchronous replies."""

    def __init__(self, parent, on_request, on_fallback, on_closed):
        self._parent = parent
        self._on_request = on_request
        self._on_fallback = on_fallback
        self._on_closed = on_closed
        self._messages = queue.Queue()
        self._reader_done = threading.Event()
        self._exit_seen = None
        self._ready = False
        self._bootstrapped = False
        self._closing = False
        self._alive = True
        self._deadline = time.monotonic() + _STARTUP_TIMEOUT
        self._process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'frontend.web_settings_host'],
            cwd=str(_ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        threading.Thread(target=self._read, daemon=True).start()
        self._poll_job = parent.after(40, self._poll)

    def _read(self):
        try:
            for line in self._process.stdout:
                if len(line) > 1024 * 1024:
                    continue
                try:
                    self._messages.put(json.loads(line))
                except (ValueError, TypeError):
                    continue
        except (OSError, ValueError):
            pass
        finally:
            self._reader_done.set()

    def _send(self, payload):
        if not self._alive:
            return
        try:
            self._process.stdin.write(json.dumps(payload, ensure_ascii=True) + '\n')
            self._process.stdin.flush()
        except (OSError, ValueError, BrokenPipeError):
            pass

    def _poll(self):
        self._poll_job = None
        if not self._alive:
            return
        while True:
            try:
                msg = self._messages.get_nowait()
            except queue.Empty:
                break
            self._dispatch(msg)
        exited = self._process.poll() is not None
        if exited and self._exit_seen is None:
            self._exit_seen = time.monotonic()
        drained = self._reader_done.is_set() or (self._exit_seen is not None and time.monotonic() - self._exit_seen >= 0.5)
        if (exited and drained) or (not self._ready and time.monotonic() >= self._deadline):
            fallback = not self._ready and not self._closing
            self._finish()
            if fallback:
                self._on_fallback()
            self._on_closed()
            return
        self._poll_job = self._parent.after(40, self._poll)

    def _dispatch(self, msg):
        if not isinstance(msg, dict):
            return
        if msg.get('event') == 'closing':
            self._closing = True
            return
        request_id, method, draft = msg.get('id'), msg.get('method'), msg.get('draft')
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            return
        if not isinstance(method, str) or method not in _METHODS or (method in {'validate', 'save'} and not isinstance(draft, dict)):
            self._send({'id': request_id, 'error': 'Invalid request'})
            return
        if method == 'bootstrap':
            self._bootstrapped = True
        elif method == 'validate' and self._bootstrapped:
            self._ready = True
        def reply(result):
            self._send({'id': request_id, 'result': result})
        try:
            self._on_request(method, draft, reply)
        except Exception:
            self._send({'id': request_id, 'error': 'Request failed'})

    def winfo_exists(self):
        return self._alive

    def focus_set(self):
        self._send({'command': 'focus'})

    def _finish(self):
        if not self._alive:
            return
        self._alive = False
        if self._poll_job is not None:
            self._parent.after_cancel(self._poll_job)
            self._poll_job = None
        try:
            self._process.stdin.close()
        except (OSError, ValueError):
            pass
        def reap():
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    self._process.kill()
                except OSError:
                    pass
                self._process.wait()
        threading.Thread(target=reap).start()

    def destroy(self):
        if self._alive:
            self._finish()
            self._on_closed()


def _webview2_installed():
    if sys.platform != 'win32':
        return False
    import winreg
    key = r'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for view in (winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
            try:
                with winreg.OpenKey(hive, key, 0, winreg.KEY_READ | view) as handle:
                    version, _ = winreg.QueryValueEx(handle, 'pv')
                if isinstance(version, str) and version.strip() not in ('', '0.0.0.0'):
                    return True
            except OSError:
                pass
    return False


class _ChildBridge:
    def __init__(self, output):
        self._output = output
        self._lock = threading.Lock()
        self._pending = {}
        self._next_id = 0
        self._window = None
        self._rules_open = threading.Event()

    def _request(self, method, draft=None):
        with self._lock:
            self._next_id += 1
            request_id = self._next_id
            pending = queue.Queue(maxsize=1)
            self._pending[request_id] = pending
            self._output.write(json.dumps({'id': request_id, 'method': method, 'draft': draft}, ensure_ascii=True) + '\n')
            self._output.flush()
        try:
            response = pending.get()
            if 'error' in response:
                raise RuntimeError(response['error'])
            return response.get('result')
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def bootstrap(self):
        import webview
        if webview.renderer != 'edgechromium':
            raise RuntimeError('Edge WebView2 is required')
        return self._request('bootstrap')

    def validate(self, draft):
        return self._request('validate', draft)

    def save(self, draft):
        return self._request('save', draft)

    def manage_rules(self, theme_preset=None):
        self._rules_open.set()
        try:
            native = self._window.native
            draft = {}
            if native is not None:
                draft["settings_hwnd"] = native.Handle.ToInt64()
            if theme_preset is not None:
                draft["theme_preset"] = theme_preset
            result = self._request('manage_rules', draft)
            # The editor belongs to a different process. Activate the existing
            # WebView explicitly after that editor releases focus.
            self._window.show()
            return result
        finally:
            self._rules_open.clear()

    def _closing(self):
        try:
            with self._lock:
                self._output.write(json.dumps({'event': 'closing'}) + '\n')
                self._output.flush()
        except (OSError, ValueError):
            pass

    def close(self):
        self._window.destroy()

    def _read_parent(self):
        try:
            for line in sys.stdin:
                try:
                    msg = json.loads(line)
                    if not isinstance(msg, dict):
                        continue
                    if msg.get('command') == 'focus' and not self._rules_open.is_set():
                        self._window.show()
                        self._window.restore()
                    elif isinstance(msg.get('id'), int):
                        with self._lock:
                            pending = self._pending.get(msg['id'])
                        if pending is not None:
                            pending.put_nowait(msg)
                except (ValueError, TypeError, queue.Full):
                    continue
        finally:
            if self._window is not None:
                self._window.destroy()


def _main():
    if not _webview2_installed():
        return 1
    # Keep the wire stream separate from diagnostic output from pywebview.
    output = sys.stdout
    sys.stdout = sys.stderr
    import webview
    bridge = _ChildBridge(output)
    html = (_ROOT / 'frontend' / 'web_settings.html').read_text(encoding='utf-8')
    bridge._window = webview.create_window('Live Trans Settings', html=html, js_api=bridge,
                                          width=488, height=760, min_size=(360, 410), on_top=True)
    bridge._window.events.closing += bridge._closing
    threading.Thread(target=bridge._read_parent, daemon=True).start()
    webview.start(gui='edgechromium')
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())






