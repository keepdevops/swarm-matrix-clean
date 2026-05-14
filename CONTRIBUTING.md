# Contributing

Thanks for taking the time. This document covers what's expected of code that
lands here — written so a new contributor can be productive in 15 minutes.

## Quick start

```bash
git clone https://github.com/keepdevops/swarm-matrix-clean.git
cd swarm-matrix-clean
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio ruff
pytest                                # 42 fast tests, ~130 ms
ruff check . && ruff format --check . # lint + format gate
python run_agent.py --list-backends   # CLI smoke test
```

If everything above passes, you're set. Open a branch and start.

## What gets enforced by CI

Every push and PR runs `.github/workflows/ci.yml`. It will block merge on any
of these:

1. **Syntax**: every `.py` must parse.
2. **300-LOC rule**: no file in `backends/`, `orchestration/`, or
   `run_agent.py` may exceed 300 lines. If yours grows past that, split it —
   don't argue with the budget.
3. **Tests pass** on `{ubuntu, macos} × py{3.10, 3.11, 3.12}`.
4. **`ruff check`** clean (rules: `E F W I B UP`).
5. **`ruff format --check`** clean — run `ruff format .` before committing.
6. **`python run_agent.py --list-backends`** exits 0.

Run all six gates locally before pushing:

```bash
ruff format . && ruff check . && pytest && python run_agent.py --list-backends
```

## Non-negotiable design rules

These come from the project's modular-reliability charter. Reviewers will
push back if you break them:

- **No silent failures.** Every `try/except` must log via
  `logger.error(..., exc_info=True)`. Empty `except: pass` is rejected.
- **Schema-validated inputs.** Extend `BackendConfig` (Pydantic) when you
  add config keys; don't read `Dict[str, Any]` and hope.
- **Cancellation works.** Any new streaming path must honor
  `asyncio.CancelledError`. Subprocess backends close the upstream stream;
  in-process backends signal a `stop_flag` to their worker thread and
  `join` with a bounded timeout.
- **Air-gap safe.** No outbound network in tests. No auto-downloads. Bind
  servers to `127.0.0.1`.
- **No comments restating the code.** Only write a comment when the *why* is
  non-obvious — a hidden constraint, a workaround, a footgun. Don't narrate.
- **No new top-level docs files** unless asked; this guide and the README
  are the canonical surface.

## Adding a new backend

Every backend implements the same contract. Use an existing module as a
template — pick the closest match:

| Style                  | Template                         |
| ---------------------- | -------------------------------- |
| In-process Python      | `backends/llama_python.py`       |
| In-process w/ threads  | `backends/mlx_engine.py`         |
| Subprocess server      | `backends/llama_cpp.py`          |
| OpenAI-compatible API  | `backends/vllm_engine.py`        |
| HF transformers-style  | `backends/transformers_engine.py`|

Checklist:

- [ ] Subclass `InferenceBackend` with `name = "<your_key>"`.
- [ ] Implement `initialize`, `generate_stream`, `shutdown`. Fail loudly on
      misconfiguration; return `False` from `initialize` (don't raise).
- [ ] Defer heavy imports (`from your_lib import ...`) into `initialize()`
      so the module is importable on hosts without the engine installed.
- [ ] Yield chunks in the project's OpenAI-shaped form:
      `{"choices": [{"delta": {"content": str}, "finish_reason": str|None}]}`.
- [ ] Add the engine's optional pip line to `requirements.txt` (commented).
- [ ] Register the backend in `orchestration/manager.py` `_registry`.
- [ ] Add a sample config under `config/agents/<role>.json`.
- [ ] Add a row to the backend matrix in `README.md`.
- [ ] Add an opt-in integration test in
      `tests/test_integration_real_models.py` keyed by a new env var.

## Tests

- Fast suite: pure unit + manager + CLI tests, fully hermetic via
  `FakeBackend`. Default `pytest` runs only these.
- Integration suite: marked `@pytest.mark.integration`, opt-in via
  `pytest -m integration`. Skip-loudly on missing deps or env vars; never
  hit the network.
- New behavior gets a test in the same PR. Bug fixes get a regression test
  that fails before the fix and passes after.

## Pull requests

- Branch off `main`. Keep PRs scoped — one concern per PR.
- **Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `ci:`.
  Subject ≤ 70 chars; explain the *why* in the body if non-obvious.
- PR description: bullet summary + a test plan checklist. The reviewer
  needs to know what to verify.
- Don't squash other people's commits without asking.
- Never force-push to `main`.

## Reporting issues

Open a GitHub issue with:

1. What you were trying to do.
2. Backend (`backend_target`) and engine version.
3. Minimal config + prompt that reproduces it.
4. Full log with `python run_agent.py ... -vv` (DEBUG).

Redact model paths if they're sensitive.

## License

By contributing you agree your work is offered under the project's
[Apache 2.0 license](LICENSE).
