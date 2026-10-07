# vehicle-splat

Turns a van's Imagin renders into a 3D Gaussian splat that the customer portal can
turn to any angle. Give it a CAP id. It shows you Imagin's pictures of the van, and
then trains the splat on your GPU.

It runs on one machine, for one person. There is no sign-in, so it listens on
localhost only.

Picking this up, or setting it up for the first time? [HANDOFF.md](HANDOFF.md)
says what has been proven, what hasn't, and what to check on the first GPU run.

## What it makes

For CAP id 58438 (or `58438-<paintId>` when you give a paint), a job writes:

| File | What it is |
|---|---|
| `58438.splat` | The model in the common web splat format, 32 bytes a splat. Viewers such as [SuperSplat](https://superspl.at/editor) open it as it is. |
| `58438.json` | The manifest the portal reads first. It names the `.splat` and gives the spin's camera, so the model is framed exactly as Imagin's renders are. |
| `58438.ply` | The standard 3D Gaussian Splatting PLY, with full spherical harmonics, for desktop tools. |
| `58438-turntable.mp4` | A preview: one turn at the spin's height, then one from above. |

The portal's renderer is in the CRM repo, under
`packages/ui/src/vehicleSpin/splat/`. To show a model there, copy the `.json` and
`.splat` into `apps/portal/public/models/`, or let the page do it (see
[Publishing to the portal](#publishing-to-the-portal)).

## How it works

`src/vehicle_splat/pipeline.py` runs the stages in order:

1. **Frames.** It fetches the 32-frame spin (Imagin angles 200–231) and Imagin's
   other views (codes 1–33). Imagin answers a code a van lacks with its default
   view, byte for byte, so those repeats are dropped and about 13 views are left.
   Every render is cached under `data/imagin/`.
2. **Camera.** Imagin doesn't publish the spin's camera, so it is fitted. For each
   guess at field of view, elevation and turning direction, the program carves the
   van's visual hull from the 32 outlines. It keeps the guess whose hull best fills
   every outline (`hull.py`).
3. **Views.** Each extra view is placed against that hull, by its outline
   (`views.py`). An outline can't tell a van's front-left from its rear-right, so a
   first model is trained on the spin alone. It renders both candidates, and the
   one closer to Imagin's picture wins. By default only the views from clearly
   above are used, because they show the roof, which the spin never sees.
4. **Training.** gsplat trains on the spin and the chosen views, with MCMC
   densification capped at "most splats". The extra views' cameras are refined
   as it goes (`train.py`).
5. **Preview** and 6. **Export** (`preview.py`, `export.py`).

The camera conventions are in `camera.py`. The portal rebuilds the same camera from
the manifest, and `tests/test_camera.py` pins the matrix that both sides are tested
against.

## Running it on Windows with WSL and an NVIDIA GPU

### 1. The GPU in WSL

You need three things:

- **The NVIDIA driver on Windows.** This is the ordinary Game Ready or Studio
  driver. Don't install a Linux driver inside WSL: WSL passes the Windows one
  through.
- **Docker with GPU support.** Either of these works:
  - Docker Desktop with WSL 2 integration turned on for your Ubuntu distribution
    (Settings → Resources → WSL integration). It includes GPU support.
  - Docker Engine installed inside Ubuntu, plus the
    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

    ```bash
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo service docker restart
    ```

- **A check.** Run this inside Ubuntu. It should print your RTX 3080:

  ```bash
  nvidia-smi
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

### 2. Start it

From this folder in Ubuntu:

```bash
mkdir -p data                 # before the first start, so the folder is yours rather than root's
docker compose up -d --build
```

The first build downloads about 6 GB: CUDA's runtime, PyTorch and gsplat's prebuilt
kernels. Nothing is compiled, and later builds after a code change take seconds.

Open **http://localhost:8000** in a browser on Windows (WSL forwards the port). The
chip at the top names the GPU. If it says no GPU is visible, go back to step 1.

```bash
docker compose logs -f        # follow what it's doing
docker compose down           # stop it; jobs and the cache stay in ./data
```

### 3. Make a splat

1. Enter a CAP id and, if you want, a paint id, then choose **Show the van**. The
   page shows the spin and Imagin's other views. Codes that only repeat the default
   view are dimmed, and the views expected from above are badged.
2. Choose the extra views, the training steps and the most splats, then choose
   **Generate splat**. The job lists its stages as they run, shows a progress bar
   while it trains, and ends with the turntable video, the files and how each view
   was placed.

Only one job trains at a time, because there is one GPU. Others queue behind it. A
job that was running when the service stopped is marked failed when it next starts,
and one still queued is run.

On an RTX 3080, estimated rather than measured (the page shows each stage's real
time as it finishes):

| Stage | Time |
|---|---|
| Fitting the camera | 1–3 min |
| Placing the views and the first model | 2–4 min |
| Training, 15,000 steps | 8–15 min |

With **None: the spin alone**, the views stage is skipped.

## Publishing to the portal

To let **Publish to the portal** copy a finished model straight into a CRM
checkout, uncomment the two lines in `compose.yaml` that mount the portal's
`apps/portal/public/models` and set `PORTAL_MODELS_DIR`. Then run
`docker compose up -d` again. Without them, the button is hidden and you copy the
files by hand.

## Settings

These are environment variables. `compose.yaml` sets them for the container.

| Variable | Default | |
|---|---|---|
| `DATA_DIR` | `data` (`/data` in the image) | Jobs, their files and Imagin's cached renders. |
| `IMAGIN_CUSTOMER` | `uk-vanaways` | Imagin's customer key, as the portal uses it. |
| `PORTAL_MODELS_DIR` | unset | Where **Publish to the portal** copies a model. Unset hides the button. |

## From the command line

The same pipeline runs without the page, for example inside the container:

```bash
docker compose run --rm vehicle-splat python3 -m vehicle_splat run --cap-id 58438 --out /data/out
```

## Developing

The tests need neither a GPU nor gsplat. They run on PyTorch's CPU build, with
Imagin and training replaced by fakes:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/pytest
```

To work on the page without a GPU, `.venv/bin/python -m vehicle_splat serve` runs it
locally. A job then fails as soon as it starts, saying no GPU is visible.
