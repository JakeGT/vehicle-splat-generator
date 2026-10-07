"""`python -m vehicle_splat serve` runs the page and API; `python -m vehicle_splat run
--cap-id 58438` makes one van's splat from the command line, into --out."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


class PrintingReporter:
    def stage(self, name, status, detail=None):
        print(f"[{name}] {status}{': ' + detail if detail else ''}", flush=True)

    def training(self, step, steps, loss, splats):
        if step % 1000 == 0 or step == steps:
            print(f"[training] step {step:,}/{steps:,}  loss {loss:.4f}  splats {splats:,}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vehicle-splat")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="the page and API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    run = commands.add_parser("run", help="one van's splat, from the command line")
    run.add_argument("--cap-id", type=int, required=True)
    run.add_argument("--paint-id")
    run.add_argument("--extra-views", choices=("above", "all", "none"), default="above")
    run.add_argument("--steps", type=int, default=15_000)
    run.add_argument("--max-splats", type=int, default=150_000)
    run.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out"))
    arguments = parser.parse_args(argv)

    if arguments.command == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(), host=arguments.host, port=arguments.port)
        return 0

    from .config import load_settings
    from .imagin import Imagin
    from .pipeline import JobSpec, run as run_pipeline

    settings = load_settings()
    spec = JobSpec(cap_id=arguments.cap_id, paint_id=arguments.paint_id, extra_views=arguments.extra_views,
                   steps=arguments.steps, max_splats=arguments.max_splats)
    report = run_pipeline(spec, arguments.out, Imagin(settings.imagin_customer, settings.imagin_cache_dir),
                          PrintingReporter())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
