"""Unified CLI loader: run any backend through one entry point.

Examples:
    python run_agent.py --agent config/agents/developer.json \\
        --prompt "// fast inverse square root\\n"

    echo "explain this" | python run_agent.py --agent config/agents/portable.json

    python run_agent.py --list-backends
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Any

from orchestration.manager import BackendManager, load_agent_config


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )


def _read_prompt(arg_prompt: str | None, prompt_file: str | None) -> str:
    if prompt_file:
        p = Path(prompt_file)
        if not p.exists():
            raise FileNotFoundError(f"prompt file not found: {p}")
        return p.read_text(encoding="utf-8")
    if arg_prompt is not None:
        return arg_prompt
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("no prompt provided; pass --prompt, --prompt-file, or pipe via stdin")


def _merge_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """CLI flags override JSON config keys, mirroring backend kwargs."""
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.max_tokens is not None:
        cfg["max_tokens"] = args.max_tokens
    if args.model_path is not None:
        cfg["model_path"] = args.model_path
    if args.backend is not None:
        cfg["backend_target"] = args.backend
    return cfg


async def _run(args: argparse.Namespace) -> int:
    logger = logging.getLogger("run_agent")
    cfg = load_agent_config(args.agent)
    cfg = _merge_overrides(cfg, args)

    prompt = _read_prompt(args.prompt, args.prompt_file)
    target = cfg.get("backend_target", "?")
    logger.info("loading backend %s (model=%s)", target, cfg.get("model_path"))

    mgr = BackendManager(max_resident=args.max_resident)

    # Forward SIGINT to a cancellation event so we can shut down gracefully.
    cancel_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, cancel_event.set)
        except (NotImplementedError, RuntimeError):
            # Windows / non-main-thread fallback
            pass

    try:
        backend = await mgr.acquire(cfg)
    except Exception:
        logger.error("backend acquisition failed", exc_info=True)
        await mgr.release_all()
        return 2

    payload: dict[str, Any] = {"prompt": prompt}
    exit_code = 0
    stream_task = asyncio.create_task(_drive_stream(backend, payload))
    cancel_task = asyncio.create_task(cancel_event.wait())

    try:
        done, pending = await asyncio.wait(
            {stream_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if cancel_task in done and not stream_task.done():
            logger.info("cancellation requested; stopping stream")
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.error("stream raised during cancel", exc_info=True)
            exit_code = 130
        else:
            cancel_task.cancel()
            result = stream_task.result()
            if result is False:
                exit_code = 1
    finally:
        await mgr.release_all()

    sys.stdout.write("\n")
    sys.stdout.flush()
    return exit_code


async def _drive_stream(backend, payload: dict[str, Any]) -> bool:
    """Pump generate_stream to stdout. Return False on yielded error chunk."""
    logger = logging.getLogger("run_agent.stream")
    ok = True
    async for chunk in backend.generate_stream(payload):
        if "error" in chunk:
            logger.error("backend error: %s", chunk["error"])
            ok = False
            break
        choice = (chunk.get("choices") or [{}])[0]
        text = (choice.get("delta") or {}).get("content", "")
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()
        if choice.get("finish_reason"):
            break
    return ok


def _list_backends() -> int:
    print("registered backends:")
    for key, cls in sorted(BackendManager._registry.items()):
        print(f"  {key:20s} -> {cls.__module__}.{cls.__name__}")
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_agent",
        description="Unified loader for matrix-safe inference backends.",
    )
    p.add_argument("--agent", "-a", help="Path to agent config JSON.")
    p.add_argument("--prompt", "-p", default=None, help="Prompt string (inline).")
    p.add_argument("--prompt-file", "-f", default=None, help="Read prompt from file.")
    p.add_argument("--backend", "-b", default=None, help="Override backend_target from the config.")
    p.add_argument("--model-path", default=None, help="Override model_path.")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument(
        "--max-resident",
        type=int,
        default=1,
        help="How many backends BackendManager keeps live (default 1).",
    )
    p.add_argument(
        "--list-backends",
        action="store_true",
        help="Print the registered backend keys and exit.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    _configure_logging(args.verbose)

    if args.list_backends:
        return _list_backends()

    if not args.agent:
        logging.error("--agent is required (or pass --list-backends)")
        return 2

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except Exception:
        logging.error("unified loader failed", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
