"""CLI smoke tests for run_agent.py."""

from __future__ import annotations

import json

import run_agent
from tests.conftest import FakeBackend


def test_list_backends_exit_zero(capsys):
    rc = run_agent.main(["--list-backends"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "registered backends" in out
    # All 6 production backends should be listed
    for key in (
        "llama_cpp_binary",
        "llama_cpp_python",
        "mlx",
        "vllm",
        "exllamav2",
        "transformers",
    ):
        assert key in out


def test_missing_agent_returns_two(caplog):
    caplog.set_level("ERROR")
    rc = run_agent.main([])
    assert rc == 2
    assert any("--agent is required" in r.message for r in caplog.records)


def test_run_against_fake_backend(tmp_path, monkeypatch, capsys):
    """Full happy-path: JSON config → BackendManager → stream → stdout."""
    from orchestration.manager import BackendManager

    monkeypatch.setattr(BackendManager, "_registry", {"fake": FakeBackend})

    cfg_path = tmp_path / "agent.json"
    cfg_path.write_text(
        json.dumps(
            {
                "backend_target": "fake",
                "tokens": ["foo", "-", "bar"],
            }
        )
    )

    rc = run_agent.main(["--agent", str(cfg_path), "--prompt", "ignored"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "foo-bar" in out


def test_cli_override_backend(tmp_path, monkeypatch, capsys):
    from orchestration.manager import BackendManager

    monkeypatch.setattr(BackendManager, "_registry", {"fake": FakeBackend})

    cfg_path = tmp_path / "agent.json"
    cfg_path.write_text(
        json.dumps(
            {
                "backend_target": "wrong_target",
                "tokens": ["x"],
            }
        )
    )

    rc = run_agent.main(
        [
            "--agent",
            str(cfg_path),
            "--backend",
            "fake",
            "--prompt",
            "p",
        ]
    )
    assert rc == 0
    assert "x" in capsys.readouterr().out


def test_cli_init_failure_returns_two(tmp_path, monkeypatch):
    from orchestration.manager import BackendManager

    monkeypatch.setattr(BackendManager, "_registry", {"fake": FakeBackend})

    cfg_path = tmp_path / "agent.json"
    cfg_path.write_text(
        json.dumps(
            {
                "backend_target": "fake",
                "force_fail": True,
            }
        )
    )

    rc = run_agent.main(["--agent", str(cfg_path), "--prompt", "p"])
    assert rc == 2


def test_prompt_file_reading(tmp_path, monkeypatch, capsys):
    from orchestration.manager import BackendManager

    monkeypatch.setattr(BackendManager, "_registry", {"fake": FakeBackend})

    cfg_path = tmp_path / "agent.json"
    cfg_path.write_text(json.dumps({"backend_target": "fake", "tokens": ["ok"]}))
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("hello from file")

    rc = run_agent.main(
        [
            "--agent",
            str(cfg_path),
            "--prompt-file",
            str(prompt_path),
        ]
    )
    assert rc == 0
    assert "ok" in capsys.readouterr().out


def test_no_prompt_no_stdin_raises(tmp_path, monkeypatch, capsys):
    """When TTY is detected and no prompt given, asyncio.run should surface
    the error via the top-level except → exit 1."""
    from orchestration.manager import BackendManager

    monkeypatch.setattr(BackendManager, "_registry", {"fake": FakeBackend})
    # Force isatty() to True so stdin path isn't taken.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    cfg_path = tmp_path / "agent.json"
    cfg_path.write_text(json.dumps({"backend_target": "fake"}))

    rc = run_agent.main(["--agent", str(cfg_path)])
    assert rc == 1
