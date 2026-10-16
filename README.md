# quadtime

Computer-vision timing for FPV drone racing, from the drone's own camera.

quadtime watches a first-person video feed, uses the markers on a gate to work
out how far the drone is from the gate plane, and scores a lap the moment it
crosses. It serves a small Flask web UI with Start/Stop/Reset buttons, a 5
second countdown, a live timer and a lap list.

This is a **proof of concept**: one pilot, one gate (start/finish), open-ended
race format, FPV camera only.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# run the built-in synthetic fly-through - no camera or hardware needed
.venv/bin/python app.py
```

Open <http://127.0.0.1:5000/>, press **Start**, and after the 5 second countdown
laps begin appearing. The synthetic scene flies a virtual camera through a gate
built with the configured real dimensions and marker layout, so the whole
pipeline is exercised end to end. Every pass is different - the drone lines up
from a different distance and direction, crosses somewhere else in the opening,
banks into the turn and takes a different time to come back around - so the lap
times vary the way they do in a real race.

Then point it at the real downlink:

```bash
.venv/bin/python app.py --source 0            # video receiver into a capture card
.venv/bin/python app.py --source 1 --fps 25   # PAL downlink
.venv/bin/python app.py --source clip.mp4     # replay a recording
```

## Why this is not just "detect the drone"

With a static camera beside the gate, timing is easy: find the drone blob and
watch it cross a line. With the camera *on the drone* that is impossible - the
drone is never in frame, and the whole scene moves. So the roles invert: the
gate is the target, and the measurement is how far the camera is from the gate
plane.

## The gate

The gate carries four **AprilTag 36h11** markers, one flat on each inside
corner, mounted upright. AprilTags are more tolerant of motion blur than ArUco,
and distinct IDs mean the gate can be identified from any angle.

```bash
.venv/bin/python tools/make_markers.py            # 75 mm markers by default
```

This writes one PDF per marker - `markers/marker_0.pdf` and so on - with each
marker drawn as vector artwork at its true physical size on its own A4 page.
There is no browser in the loop and nothing to scale: print the page on A4 at
100% (leave "fit to page" off) and the black square comes out at 75 mm. Each
page also has a 50 mm ruler, so the scale can be checked with a rule before the
marker goes near the gate. Keep the white border clear.

```bash
.venv/bin/python tools/make_markers.py --marker-size-mm 40 --page letter
```

| marker ID | inside corner |
|-----------|---------------|
| 0         | top-left      |
| 1         | top-right     |
| 2         | bottom-right  |
| 3         | bottom-left   |

The default gate is a **24 inch (0.6096 m) square opening**. Both the gate size
and the marker size are configuration - see `gate_width_m`, `gate_height_m` and
`marker_size_m`.

### 3D printed markers

To print the markers instead of sticking paper on the gate:

```bash
.venv/bin/python tools/make_markers.py --format stl
```

This writes two layers per marker - `marker_<id>_base.stl` and
`marker_<id>_pattern.stl` - in millimetres and in the same coordinate frame, so
they line up when imported together:

- **base** is the white tile: the marker square plus its quiet zone, 1.6 mm
  thick by default, sitting on the bed.
- **pattern** is the black layer: the marker's black modules, 0.6 mm thick,
  raised on top of the base. Where the marker is white you see the white base
  underneath.

Then either:

- **two materials** (MMU, AMS, dual extruder): import both files as separate
  objects and assign white and black; or
- **one extruder**: import both and let the slicer print them layer by layer,
  and add a filament change at the base thickness so the tile turns from white
  to black exactly where the pattern begins.

`--base-thickness-mm`, `--marker-thickness-mm` and `--quiet-zone-mm` adjust the
layers. The default 1.6 mm and 0.6 mm are whole numbers of 0.2 mm layers, so the
colour change falls on a layer boundary. `--format both` writes the PDFs and the
STLs together.

For a poor analogue feed: print matte, not glossy (specular highlights break the
marker pattern), and keep the markers clean and unscuffed. Retroreflective
markers plus an LED on the drone are possible but add weight and power.

## Camera calibration

Pose estimation needs the camera's intrinsics. Calibrate once per camera with a
ChArUco board:

```bash
.venv/bin/python tools/calibrate_camera.py --video calibration.mp4 --out config/camera.json
```

Wave the board slowly over the whole field of view, at a range of distances and
angles. A dozen or more good views give a solid result; the tool prints the
reprojection error when it finishes.

If `config/camera.json` is missing, quadtime falls back to a nominal pinhole
model derived from `hfov_deg` (120 degrees by default). That is enough to get
going, but a real calibration is noticeably better.

## How it works

```
downlink ─▶ capture thread ─▶ AprilTag detection ─▶ solvePnP pose
                                                        │
                                          camera position in gate coordinates
                                                        │
                                    plane-distance trajectory (fit / interpolate)
                                                        │
                                              race state machine ─▶ Flask UI / API
```

1. **Marker detection** (`quadtime/pose.py`) finds the AprilTags, tuned for
   small, blurry markers: a wide adaptive-threshold search, subpixel corner
   refinement and a higher perspective-removal resolution.
2. **Pose estimation.** The markers' known 3D positions on the gate are matched
   against the detected corners and `cv2.solvePnP` recovers the gate's pose in
   camera coordinates. At least two markers are required - a single marker is a
   poorly conditioned planar target and gives badly wrong positions.
   The planar solution ambiguity is broken by preferring the pose in which the
   gate faces the camera, and by continuing from the previous frame.
3. **Distance to the plane.** From the pose we compute the camera's position in
   gate coordinates. Its `z` component is the signed distance to the gate plane:
   positive on the approach side, zero in the plane, negative once through. Its
   sign flip is the pass.
4. **Pass detection** (`quadtime/fpv.py`) keeps a short history of that
   distance and recovers the crossing by interpolating between frames when the
   gate is visible across the plane, or by fitting a local straight line and
   extrapolating when it is not (see below).
5. **Race state machine** (`quadtime/race.py`) moves idle → countdown → running →
   stopped. Passes only score while running. Once stopped the UI shows the total
   time, the best and average lap, and a **Reset** button clears them back to
   idle so the next race starts from zero.

### Why every pass is extrapolated

At the moment the camera reaches the gate plane it is *inside* the opening, and
the corner markers are already outside the field of view - the true crossing
frame is never observed. So on approach the tracker fits the last few poses and
predicts where the distance reaches zero, a few tens of milliseconds ahead.

That prediction is guarded so it degrades to *no event* rather than a false lap:

- the crossing point must be **inside the gate opening** (so flying *past* the
  gate, which crosses the infinite plane, does not score),
- the drone must be closing faster than `min_closing_speed_mps` (so hovering
  through does not score),
- the last sighting must be closer than `close_distance_m`,
- the fit must be reasonably straight (`max_fit_residual_m`) and the prediction
  within `max_extrapolation_s`,
- then a minimum lap time and a re-arm guard prevent double counting.

## Configuration

Copy `config.example.json`, edit, and pass it with `--config`:

```bash
.venv/bin/python app.py --config my_config.json
```

The settings most worth tuning:

| key | meaning |
|-----|---------|
| `source` | `"synthetic"`, a capture-card index, or a file path |
| `gate_width_m`, `gate_height_m`, `marker_size_m` | the real gate geometry |
| `intrinsics_file`, `hfov_deg` | camera model, calibrated or nominal |
| `min_markers` | markers required for a pose (2 by default; raise it if poses are noisy) |
| `min_closing_speed_mps` | reject slow passes |
| `close_distance_m` | how near the gate must be before extrapolating |
| `min_lap_time_s` | debounce between laps |

`GET /api/config` returns the live settings.

## HTTP API

| Method | Route | Purpose |
|--------|-------|---------|
| POST | `/api/race/start` | begin the countdown |
| POST | `/api/race/stop` | end the race, return the summary |
| POST | `/api/race/reset` | clear the finished race, only once stopped |
| GET | `/api/race/state` | phase, clock, laps, gate lock and diagnostics |
| GET | `/api/race/laps` | just the lap list |
| GET | `/api/config` | current configuration |
| GET | `/stream.mjpg` | annotated MJPEG preview |

## Tools

```bash
# print the gate markers, one PDF per marker (--format stl for printable tiles)
.venv/bin/python tools/make_markers.py

# calibrate the camera
.venv/bin/python tools/calibrate_camera.py --video calibration.mp4

# run the vision pipeline over a recording and report passes
.venv/bin/python tools/analyze_clip.py clip.mp4 --out annotated.mp4
```

`analyze_clip.py` is the fastest way to tune: it reports how often the gate was
locked, the distance range over which markers were visible, every pass and the
resulting lap times.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

Covers the gate geometry, pose recovery from a known camera position, the
crossing fit and all of its rejection guards, the calibration pipeline (on
synthetic ChArUco views), the marker PDFs (module pattern, exact physical size,
page geometry), the marker STLs (volume, winding, layer alignment, and a
top-down render that still decodes as the right marker), the synthetic flight
itself (camera orientation, slanted passes, schedule variety and a check that
every generated pass scores exactly one lap) and the full race flow on the
synthetic fly-through.

## Project layout

```
app.py                    Flask app and CLI
quadtime/config.py        all tunable settings
quadtime/capture.py       capture-card / file / synthetic frame sources
quadtime/synthetic.py     synthetic fly-through: camera path, flight schedule, renderer
quadtime/gatemodel.py     the metric gate (marker 3D positions)
quadtime/pose.py          AprilTag detection and solvePnP pose
quadtime/fpv.py           plane-distance tracking and pass detection
quadtime/calibration.py   camera intrinsics, loading and nominal fallback
quadtime/race.py          FPV engine, race state machine and capture thread
quadtime/markers.py       printable marker pages and 3D printed marker tiles
quadtime/pdfcanvas.py     tiny dependency-free vector PDF writer
quadtime/stl.py           tiny dependency-free binary STL writer
templates/, static/       web UI
tools/                    marker generator, camera calibration, clip analyser
tests/                    unit and integration tests
```

## Accuracy and limitations

- **The crossing frame is never observed**, so every pass is a fitted
  zero-crossing. Accuracy depends on how many frames the markers stay visible
  approaching the gate.
- **The detection window is short.** With 75 mm markers, a 120 degree lens at
  640x480 and a 0.61 m gate, markers are detected from roughly **0.9 m down to
  0.25 m** - about **7 frames at 30 fps**, or 4 at 6 m/s. That is enough for the
  fit, but there is not much margin. Narrower-FOV lenses, higher capture
  resolution or larger markers all widen the window; a slower approach does too.
- **Downlink latency is not compensated.** Analogue/digital video transmitters
  add a roughly constant delay, which offsets absolute times against a ground
  start but cancels out of lap-to-lap times. A `latency_compensation_ms` option
  is a natural next step.
- **Motion blur and rolling shutter** degrade marker detection exactly when the
  drone is fastest. Keep the exposure short.
- **Variable frame rate** is handled - the fit uses real frame timestamps, not a
  fixed interval.
- **One gate, one pilot.** Multiple gates would need the marker IDs decoded into
  gate identities.

### Possible next steps

- Downlink latency calibration against a second timer.
- High-contrast stripes on the gate's edges, which stay visible closer to the
  crossing than the corner markers and could corroborate the extrapolation.
- Multiple gates and pilots.
- A lap-count race format with a countdown to finish.
