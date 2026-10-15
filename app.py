"""quadtime web application.

quadtime times FPV drone races from a first-person camera feed: it uses the
gate markers to recover the drone's 3D distance to the gate plane and scores a
lap when the drone crosses it.

Run with::

    python app.py                      # synthetic fly-through, no hardware
    python app.py --source 0           # first camera / capture card
    python app.py --source clip.mp4    # replay a recorded clip
"""

from __future__ import annotations

import argparse
import atexit

from flask import Flask, Response, jsonify, render_template

from quadtime.config import Config, load_config
from quadtime.race import RaceController


def create_app(cfg: Config) -> tuple[Flask, RaceController]:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    controller = RaceController(cfg)
    controller.launch()
    atexit.register(controller.shutdown)

    @app.get("/")
    def index():
        return render_template("index.html", cfg=cfg.to_dict())

    @app.post("/api/race/start")
    def race_start():
        controller.start_race()
        return jsonify(controller.snapshot())

    @app.post("/api/race/stop")
    def race_stop():
        controller.stop_race()
        return jsonify(controller.snapshot())

    @app.get("/api/race/state")
    def race_state():
        return jsonify(controller.snapshot())

    @app.get("/api/race/laps")
    def race_laps():
        return jsonify(controller.snapshot()["laps"])

    @app.get("/api/config")
    def show_config():
        return jsonify(cfg.to_dict())

    @app.get("/stream.mjpg")
    def stream():
        return Response(
            controller.mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app, controller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quadtime", description="FPV race timing")
    parser.add_argument("--config", help="JSON config file overriding defaults")
    parser.add_argument(
        "--source",
        help="'synthetic', a camera index such as 0, or a video file path",
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config) if args.config else Config()
    for name in ("source", "width", "height", "fps", "host", "port"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)

    app, _controller = create_app(cfg)
    print(f"quadtime: source={cfg.source!r}  http://{cfg.host}:{cfg.port}/")
    app.run(
        host=cfg.host,
        port=cfg.port,
        threaded=True,
        debug=args.debug,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
