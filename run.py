#!/usr/bin/env python3
"""Local, single-user TorchRL exercise runner. Submitted code runs as you."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

PUBLIC_ORIGIN = "https://torchrl-learning-lab.naistfreib.chatgpt.site"
MAX_BODY = 128 * 1024
MAX_CODE = 64 * 1024
MAX_OUTPUT = 12 * 1024
RUN_TIMEOUT = 20.0
BASE = Path(__file__).resolve().parent


def versions():
    out = {"python": sys.version.split()[0]}
    for package in ("torch", "torchrl", "tensordict"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def error_text(error):
    line = None
    if isinstance(error, SyntaxError) and error.filename == "answer.py":
        line = error.lineno
    else:
        for frame in traceback.extract_tb(error.__traceback__):
            if frame.filename == "answer.py":
                line = frame.lineno
    location = ("answer.py:" + str(line) + ": ") if line else ""
    return (location + type(error).__name__ + ": " + str(error))[:1800]


def run_worker(payload, result_path):
    """Not a sandbox. Isolation prevents exercise state leaking between runs."""
    tests = payload["tests"]
    answer_namespace = {"__name__": "__exercise__", "__file__": "answer.py"}
    results = []
    try:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)
        exec(compile(payload["code"], "answer.py", "exec"), answer_namespace)
    except BaseException as error:
        results = [{"name": test["name"], "passed": False,
                    "error": "解答の読み込み: " + error_text(error)} for test in tests]
    else:
        test_namespace = {"__name__": "__exercise_tests__"}
        try:
            exec(compile(payload.get("setup", ""), "setup.py", "exec"), test_namespace)
            # User functions retain answer_namespace as their globals. Test imports
            # must never make omitted imports in the submitted answer work.
            test_namespace.update(answer_namespace)
        except BaseException as error:
            results = [{"name": test["name"], "passed": False,
                        "error": "テスト準備: " + error_text(error)} for test in tests]
        else:
            for test in tests:
                try:
                    exec(compile(test["code"], "test.py", "exec"), test_namespace)
                except BaseException as error:
                    results.append({"name": test["name"], "passed": False,
                                    "error": error_text(error)})
                else:
                    results.append({"name": test["name"], "passed": True, "error": None})
    Path(result_path).write_text(json.dumps({"results": results}, ensure_ascii=False), encoding="utf-8")


def _capture(stream, chunks):
    remaining = MAX_OUTPUT
    total = 0
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        total += len(chunk)
        if remaining:
            kept = chunk[:remaining]
            chunks.append(kept)
            remaining -= len(kept)
    if total > MAX_OUTPUT:
        chunks.append("\n…出力を省略しました。\n".encode())
    stream.close()


def execute(code, setup, tests, timeout=RUN_TIMEOUT):
    started = time.monotonic()
    payload = {"code": code, "setup": setup, "tests": tests}
    captured = [[], []]
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="torchrl-answer-") as folder:
        result_path = str(Path(folder) / "result.json")
        kwargs = {"start_new_session": True} if os.name == "posix" else {}
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", result_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=folder, env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}, **kwargs)
        readers = [threading.Thread(target=_capture, args=(stream, output), daemon=True)
                   for stream, output in zip((process.stdout, process.stderr), captured)]
        for reader in readers:
            reader.start()
        try:
            process.stdin.write(json.dumps(payload).encode())
            process.stdin.close()
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
        finally:
            for reader in readers:
                reader.join(timeout=1)
        if timed_out:
            result = {"results": [{"name": item["name"], "passed": False,
                                  "error": "実行時間の上限を超えました。ループや計算量を確認してください。"}
                                 for item in tests]}
        else:
            try:
                result = json.loads(Path(result_path).read_text(encoding="utf-8"))
                if not isinstance(result.get("results"), list):
                    raise ValueError("Invalid result")
            except (OSError, ValueError):
                result = {"results": [{"name": item["name"], "passed": False,
                                      "error": "Python が結果を返さずに終了しました。エラー出力を確認してください。"}
                                     for item in tests]}
    result.update(stdout=b"".join(captured[0]).decode("utf-8", "replace"),
                  stderr=b"".join(captured[1]).decode("utf-8", "replace"),
                  timed_out=timed_out, duration_ms=round((time.monotonic() - started) * 1000),
                  code_sha256=hashlib.sha256(code.encode()).hexdigest(), versions=versions())
    return result


def load_exercises(path):
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    source = source["exercises"] if isinstance(source, dict) else source
    if not isinstance(source, list):
        raise ValueError("exercises.json must contain a list or an exercises list")
    exercises = {}
    for item in source:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("Every exercise needs a string id")
        if item["id"] in exercises:
            raise ValueError("Duplicate exercise id: " + item["id"])
        exercises[item["id"]] = item
    return exercises


def task_fixture(exercises, task_id, stage):
    item = exercises.get(task_id)
    if item is None:
        raise ValueError("この課題が実行環境にありません。学習アプリと実行環境を一緒に更新してください。")
    stages = item.get("stages")
    fixture = item
    if isinstance(stages, dict):
        if stage not in stages:
            raise ValueError("未対応の演習段階です。")
        fixture = stages[stage]
    elif isinstance(stages, list):
        fixture = next((candidate for candidate in stages
                        if isinstance(candidate, dict) and candidate.get("id") == stage), None)
        if fixture is None:
            raise ValueError("未対応の演習段階です。")
    elif stage not in ("guided", "recall", "transfer", "blank", "function", "scaffold", "practice"):
        raise ValueError("未対応の演習段階です。")
    tests = fixture.get("tests", item.get("tests", []))
    if not tests or not all(isinstance(test, dict) and isinstance(test.get("name"), str)
                            and isinstance(test.get("code"), str) for test in tests):
        raise ValueError("課題のテストが不正です。")
    return item.get("setup", "") + "\n" + (fixture.get("setup", "") if fixture is not item else ""), tests


class RunnerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, static_dir, exercises, timeout=RUN_TIMEOUT):
        self.token = secrets.token_urlsafe(32)
        self.static_dir = Path(static_dir).resolve() if static_dir else None
        self.exercises = exercises
        self.run_timeout = timeout
        self.run_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), Handler)


class Handler(SimpleHTTPRequestHandler):
    server: RunnerServer
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(args[2].static_dir or BASE), **kwargs)

    def log_message(self, format, *args):
        # Never log headers, submitted code or the session token.
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), format % args))

    def _valid_host(self):
        return self.headers.get("Host") in {
            "127.0.0.1:" + str(self.server.server_port),
            "localhost:" + str(self.server.server_port),
        }

    def _local_origins(self):
        return {"http://127.0.0.1:" + str(self.server.server_port),
                "http://localhost:" + str(self.server.server_port)}

    def _origin_ok(self, session=False):
        origin = self.headers.get("Origin")
        if origin is not None:
            return origin in self._local_origins() or (not session and origin == PUBLIC_ORIGIN)
        if session:
            return self.headers.get("Sec-Fetch-Site") == "same-origin"
        # Authenticated local command-line clients need not send Origin.
        return self.headers.get("Sec-Fetch-Site") not in ("cross-site", "same-site")

    def _json(self, status, body, cors=True):
        payload = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Vary", "Origin")
        origin = self.headers.get("Origin")
        if cors and origin in self._local_origins() | {PUBLIC_ORIGIN}:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        if not self._valid_host() or not self._origin_ok(session=self.path == "/api/session"):
            return self._json(403, {"error": "接続元が許可されていません。"}, cors=False)
        if self.path not in ("/api/status", "/api/run"):
            return self._json(404, {"error": "Not found"})
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin", ""))
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        if self.headers.get("Access-Control-Request-Private-Network") == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Vary", "Origin")
        self.end_headers()

    def do_GET(self):
        if not self._valid_host():
            return self._json(403, {"error": "Host が許可されていません。"}, cors=False)
        path = urlsplit(self.path).path
        if path == "/api/session":
            if not self._origin_ok(session=True):
                return self._json(403, {"error": "実行用のローカル学習ページを開いてください。"}, cors=False)
            return self._json(200, {"token": self.server.token, "versions": versions()}, cors=False)
        if path == "/api/status":
            if not self._origin_ok():
                return self._json(403, {"error": "接続元が許可されていません。"}, cors=False)
            version_info = versions()
            return self._json(200, {"ready": all(version_info.get(name) for name in ("torch", "torchrl", "tensordict")),
                                    "authenticated": hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + self.server.token),
                                    "versions": version_info, "exercise_count": len(self.server.exercises)})
        if path.startswith("/api/"):
            return self._json(404, {"error": "Not found"})
        if self.server.static_dir is None:
            return self._json(404, {"error": "--static-dir で学習アプリのフォルダーを指定してください。"})
        return super().do_GET()

    def do_HEAD(self):
        if not self._valid_host() or self.server.static_dir is None or urlsplit(self.path).path.startswith("/api/"):
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        return super().do_HEAD()

    def list_directory(self, path):
        self.send_error(404, "Not found")
        return None

    def do_POST(self):
        # All POST replies close the connection, including rejected oversized bodies.
        self.close_connection = True
        if not self._valid_host() or not self._origin_ok():
            return self._json(403, {"error": "接続元が許可されていません。"}, cors=False)
        if urlsplit(self.path).path != "/api/run":
            return self._json(404, {"error": "Not found"})
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + self.server.token):
            return self._json(401, {"error": "実行環境に接続し直してください。"})
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            return self._json(415, {"error": "JSON で送信してください。"})
        if self.headers.get("Transfer-Encoding"):
            return self._json(400, {"error": "Content-Length が必要です。"})
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            return self._json(413, {"error": "送信内容が大きすぎます。"})
        self.connection.settimeout(5)
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete body")
            data = json.loads(raw)
            if not isinstance(data, dict) or set(data) - {"id", "stage", "code"}:
                raise ValueError("id, stage, code のみ送信できます。")
            if not all(isinstance(data.get(key), str) for key in ("id", "stage", "code")):
                raise ValueError("id, stage, code は文字列で指定してください。")
            if len(data["code"].encode()) > MAX_CODE:
                raise ValueError("解答は 64 KB 以内にしてください。")
            setup, tests = task_fixture(self.server.exercises, data["id"], data["stage"])
        except (ValueError, UnicodeDecodeError, TimeoutError) as error:
            return self._json(400, {"error": str(error)[:300]})
        if not self.server.run_lock.acquire(blocking=False):
            return self._json(409, {"error": "前の実行が終わるまでお待ちください。"})
        try:
            result = execute(data["code"], setup, tests, timeout=self.server.run_timeout)
            return self._json(200, result)
        except Exception as error:
            return self._json(500, {"error": "実行環境のエラー: " + error_text(error)})
        finally:
            self.server.run_lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--static-dir", type=Path, default=BASE / "site")
    parser.add_argument("--exercises", type=Path, default=BASE / "exercises.json")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(json.load(sys.stdin), args.worker)
        return
    try:
        exercises = load_exercises(args.exercises)
    except (OSError, ValueError) as error:
        parser.error("課題を読み込めません: " + str(error))
    server = RunnerServer(args.port, args.static_dir, exercises)
    print("学習ページ: http://127.0.0.1:" + str(server.server_port) + "/", flush=True)
    print("終了: Ctrl+C。入力した Python はこの PC の権限で実行されます。", flush=True)
    print("公開ページを接続する場合のトークン: " + server.token, flush=True)
    print("ライブラリ: " + json.dumps(versions(), ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
