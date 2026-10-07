# Handoff

Where the service stands, what is known about it and what is next. The
[README](README.md) covers what it is and how to run it. This file covers what
the README doesn't: what has and hasn't been proven, why it is built the way it
is, and the facts it depends on.

*Written 7 October 2026. The code was written and tested on a Linux machine
without an NVIDIA GPU. It hasn't been run on one yet.*

## Status

| | |
|---|---|
| Pipeline, API, page | Written. |
| Tests | 56 pass, on PyTorch's CPU build, in about 20 seconds. |
| The page | Driven end to end in headless Chrome, against a stand-in Imagin and a stand-in runner. |
| Docker image | Not built yet. The base image, the torch wheel and the gsplat wheel it pins are confirmed to exist for Python 3.10. |
| Training on a GPU | **Not run yet.** gsplat has never been imported here, and no real van has been trained by this code. |

This service started as a Colab notebook. Only the notebook's CPU parts have been
run: on CAP 58438 (a Ford Transit), they fitted the camera and built the starting
splats. Exported as a model, those starting splats lined up with Imagin's frames
in the portal at every angle tested (outline IoU 0.93–0.96). That proves the
camera convention end to end. The notebook's training was never run, and neither
has this service's. So the first GPU run tests everything from training on, plus
what the service adds: Imagin's views from above, choosing which to use, and
refining their poses.

## First run on the RTX 3080 machine

1. Follow the README's "Running it on Windows with WSL and an NVIDIA GPU" to the
   point where `docker compose up -d --build` succeeds. The chip at the top of
   http://localhost:8000 should name the RTX 3080.
2. **CAP 58438, "None: the spin alone".** This is the simplest run: the spin and
   training, and nothing else. Expect a camera fit of about **field of view 9°,
   elevation 1°, turning −, hull fit 0.94**. That is what the notebook found for
   this van, and a very different camera means the fit has gone wrong. The
   turntable video should look like the van from every angle at the spin's
   height.
3. **CAP 58438, "From above"** (the default). Check:
   - **The views table.** Codes 17, 21, 27 and 33 should place at about 19°, 15°,
     31° and 90° and be used. The others should be "at the spin's own height".
   - **The second ring of the video**, the one from above. The roof should be a
     roof now, not a guess. If it is smeared or doubled, a view was placed or
     flipped wrongly. The `alternative` in the job's `report.views` (in
     `data/jobs/<id>/job.json`) shows what the other candidate scored.
4. Write down each stage's time, which the page shows. The README's timings are
   estimates and should be replaced with these.
5. To see it in the portal, copy the `.json` and `.splat` into the CRM's
   `apps/portal/public/models/`. Imagin's frames show when the spin rests and the
   model shows while it turns, so the two should line up exactly.

### If it fails

| Symptom | Where to look |
|---|---|
| "No GPU visible" on the chip | Docker can't see the GPU. Run the README's `nvidia-smi` check inside a container. In WSL, use the Windows driver only, and never install a Linux one. |
| Build fails at the gsplat line | Its wheel index is `https://docs.gsplat.studio/whl/pt24cu124`. The pins must stay together: PyTorch 2.4.1 with CUDA 12.4 needs the `+pt24cu124` build, on Python 3.10, which is why the base is Ubuntu 22.04. |
| `ImportError` from gsplat at the first training step | Usually a torch/gsplat mismatch, as in the row above. gsplat is only imported when training starts, so the page and the camera fit run without it. |
| An error from `MCMCStrategy` | `train.py` follows gsplat 1.5.3's API: `initialize_state()` with no arguments, then `step_post_backward(..., lr=)`. Other gsplat versions differ. |
| CUDA out of memory | Lower *most splats*. The 3080 has 10 GB, and 150,000 splats at 1200 × 750 should fit easily. |
| Files in `./data` owned by root | `./data` didn't exist before the first `docker compose up`. Run `sudo chown -R $USER data`. |
| A job sits at "queued" | Only one job runs at a time. `docker compose logs -f` shows the one that is running. |

## How it fits together

| File | Does |
|---|---|
| `imagin.py` | Fetches renders, caches them by van, width and angle, and marks the codes that only repeat the default view. |
| `camera.py` | The camera convention. The portal must build identical matrices. |
| `hull.py` | Fits the spin's camera by visual hull, and gives training its first splats. |
| `views.py` | Places Imagin's other views against the hull, each with a mirror-image alternative. |
| `train.py` | gsplat training, MCMC densification, and pose refinement for the extra views. |
| `preview.py`, `export.py` | The turntable video, and the `.splat`, `.ply` and manifest. |
| `pipeline.py` | Runs the stages in order, and decides which views to use. |
| `jobs.py`, `api.py`, `web/` | The queue, which keeps one job running at a time with its state in JSON files; the API; the page. |

The tests build a made-up van from two boxes and render it exactly by ray casting
(`tests/conftest.py`). The camera fit and view placement are tested against
cameras whose answers are known. `test_pipeline.py` runs every stage with
training and rendering replaced by stand-ins.

## Facts it depends on

**Imagin**
- The spin is angles 200–231: 32 frames an even step apart, from about the van's
  own height.
- The other views are codes 1–33. A van has 13 of its own: 1, 5, 9, 13, 17, 21,
  22, 23, 25, 27, 28, 29 and 33. For any other code, Imagin returns code 1's
  picture byte for byte, so the service drops repeats by hash.
- The views from above are 17 (about 19°), 21 (about 15°), 27 (about 31°) and 33
  (straight down). They frame the van differently from the spin, and from each
  other.
- Widths up to 2600 work. PNG keeps the transparent background, which is where
  the outlines come from.

**The camera**
- The convention is OpenCV's: x right, y down, z forward. The van turns about
  world y, and the camera stays 4.0 from the axis.
- Frame k is at yaw = direction × k × 360°/32.
- Pixels are square, and the principal point is at the centre.
- The CRM rebuilds it in `packages/ui/src/vehicleSpin/splat/turntableCamera.ts`.
  `tests/test_camera.py` here and `apps/portal/src/components/deals/vehicleSplat.test.ts`
  there pin the same matrix. Change both or neither.

**The files**
- **`.splat`:** 32 bytes per splat, in this order: position (3 × f32), scale
  (3 × f32), RGBA (u8: base colour, then opacity), and rotation (u8 w, x, y, z,
  each q × 128 + 128). Splats are sorted largest and most opaque first.
- **Manifest, version 1:** `{version, format: "splat", splat, splatCount,
  camera: {convention: "opencv-turntable", fovYDegrees, elevationDegrees,
  direction, distance, frameCount}, source, createdAt}`. The portal reads it in
  `packages/ui/src/vehicleSpin/splat/splatModel.ts`.

## Decisions, and why

- **Local, for one person, with no sign-in.** It is bound to 127.0.0.1. A tunnel
  was considered and dropped. Anything wider needs authentication first.
- **One worker, with jobs stored as JSON files.** There is one GPU, so a database
  or job broker would add parts without adding anything. After a restart, a job
  that was running is marked failed and a queued job runs again.
- **The camera is fitted, not assumed.** Imagin doesn't publish its camera. A
  guessed field of view puts the splat out of line with Imagin's frames, and the
  portal shows those frames whenever the spin rests.
- **Front and back are told apart by picture, not outline.** A boxy van's
  front-left outline is the mirror of its rear-right. A first model trained on the
  spin alone renders both candidates, and the candidate closer to Imagin's picture
  wins.
- **The views from above are used by default; the rest are not.** The spin
  already covers the van's own height. Views from that height add a second camera
  that can conflict with it, and add nothing the spin doesn't already show. *All
  of them* is there to compare against.
- **Only the extra views' poses are refined.** The spin's camera is fitted from 32
  consistent frames and is trusted. The extra views were placed from their outline
  alone.

## Open questions and next steps

- **The first GPU run.** Everything in [First run](#first-run-on-the-rtx-3080-machine).
- **The camera search ranges.** The field of view is searched from 4° to 40° and
  the elevation from −4° to 24° (`hull.py`). These were sized for the Transit.
  Try a small van and a tall one.
- **The thresholds.** `MIN_OUTLINE_IOU = 0.8` (in `pipeline.py`) and the
  8° margin for "from above" (in `views.py`) are first guesses, made before any
  real view had been placed.
- **Size.** 150,000 splats is about 4.8 MB, against about 2 MB for the 32 WebP
  frames. A compressed format such as SOG or `.spz` would close the gap, but the
  portal's loader would have to read it.
- **In the CRM.** The portal side is on the CRM branch
  `feature/portal/vehicle-splat` (plan task T3.1a). Models are copied in by hand
  until the API serves vehicle images (plan T2.1), at which point a model's URL
  travels as `VehicleImages.model`.
