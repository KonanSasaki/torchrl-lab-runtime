"""Exercise RPC used inside each learner's temporary Binder session."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from run import MAX_CODE, execute, load_exercises, task_fixture, versions

FIXTURES = Path(__file__).with_name('exercises.json')
EXPECTED = {'torch': '2.14.0', 'torchrl': '0.14.0', 'tensordict': '0.14.2'}

def status():
    installed = versions()
    ready = all((installed.get(name) or '').split('+')[0] == version
                for name, version in EXPECTED.items())
    return {'ready': ready, 'authenticated': True, 'versions': installed,
            'fixture_sha256': hashlib.sha256(FIXTURES.read_bytes()).hexdigest(),
            'runtime': 'binder'}

def run_answer(payload):
    if not isinstance(payload, dict):
        raise ValueError('実行する答案が不正です。')
    code = payload.get('code')
    if not isinstance(code, str) or not code.strip() or len(code.encode()) > MAX_CODE:
        raise ValueError('答案が空か、長すぎます。')
    if not all(isinstance(payload.get(k), str) for k in ('id', 'stage')):
        raise ValueError('演習を選び直してください。')
    exercises = load_exercises(FIXTURES)
    setup, tests = task_fixture(exercises, payload['id'], payload['stage'])
    return execute(code, setup, tests)
