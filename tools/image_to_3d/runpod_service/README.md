# Pixal3D multi-view on RunPod Serverless

Queue-based worker with RunPod API-key authentication, one initialized model per
warm worker, persistent textured GLB outputs and GPU scale-to-zero.
**Multi-view only.** The worker downloads only the checkpoints required by
Pixal3D multi-view inference.

The CPU contract tests pass. CUDA image compilation, real GPU inference and live
endpoint deployment still need verification; do not treat the GPU profile below
as a measured memory requirement.

## Inputs

Provide 2–8 images with matching camera metadata, using the upstream Blender/NeRF
camera-to-world convention. Each `transforms.json` frame needs a 4x4
`transform_matrix` and horizontal `camera_angle_x` in radians (or a shared angle
at the root). Image argument order must match frame order; frame zero should be
the canonical front view. The service preserves image framing when masking.

These must be consistent, posed views of one object. A collection of AI-generated
front/rear pictures does not automatically provide calibrated cameras. See the
[upstream multi-view examples](https://github.com/TencentARC/Pixal3D#multi-view-inference).
The eight-view maximum is a configurable service limit, not a claimed model limit.

## Build and image

GitHub Actions tests and builds the CUDA image on pushes to `main`, then publishes
`ghcr.io/i3oot/pixal3d-runpod-worker:latest` and a commit-SHA tag. The workflow uses
GitHub's built-in `GITHUB_TOKEN` with `packages:write`; no registry credential needs
to be committed. After first publication, set the GHCR package visibility to
public so RunPod can pull without a registry secret. Prefer a verified SHA tag or
digest for deployment.

Local build from the repository root:

```sh
docker build -f tools/image_to_3d/runpod_service/Dockerfile \
  -t pixal3d-runpod-worker .
```

The image targets **SM 8.9 / Ada**, CUDA 12.8 and PyTorch 2.7. Suitable proposed
GPUs include RTX 4090, L40S and RTX 6000 Ada. Other architectures require a rebuild
and kernel compatibility checks. Source revisions are pinned in `source-lock.json`;
transitive system/Python dependencies are not a complete reproducible lock.
The build context excludes inputs, outputs, model caches and credentials.

## Storage and GPU sizing

Only files referenced by the pinned multi-view pipeline are downloaded. The
required 3D checkpoints total **24.045 GB on disk**, including shared decoders.
Allow roughly **26–30 GB for checkpoints plus helper caches**, an estimate until
first startup is measured. An optional **50 GB network volume** would leave initial room for
the cache. Temporary container disk is an alternative without a persistent volume. The full upstream repository is about 46 GB because it also
contains single-view weights; this worker does not download those.

Disk size is not a VRAM requirement. Start with a **24 GB RTX 4090 and
`PIXAL_LOW_VRAM=1`** for a benchmark; 48 GB Ada is a fallback if measured peaks
require it. Multi-view peak VRAM has not been established here. Low-VRAM mode
retains loaded objects in CPU memory and moves stages to the GPU as needed;
ensure the worker also has enough host RAM.

As a planning estimate, RunPod lists RTX 4090 Serverless around $1.10 per running
hour and 48 GB Ada around $1.75, subject to current console pricing. Standard
network-volume storage below 1 TB is $0.07/GB/month: **50 GB is about $3.50/month**,
even with zero GPU workers. Startup, inference and idle time affect GPU cost.
Verify current [GPU pricing](https://www.runpod.io/product/serverless) and
[storage pricing](https://docs.runpod.io/storage/network-volumes) before activation.

## Endpoint setup

Deployment-specific values belong in the RunPod console's environment and
Secrets settings, not in this repository. No deployment settings file is tracked.
Use a queue-based endpoint and choose the GPU, EU placement, temporary disk size,
worker limits and idle/execution timeouts in the console.

Configure the worker with these environment variables:

| Variable | Purpose |
| --- | --- |
| `PIXAL_MODE` | Supported input mode: `multi` |
| `PIXAL_LOW_VRAM` | Enable CPU offloading when set to `1` |
| `MAX_VIEWS` | Maximum accepted view count |
| `MAX_RESOLUTION` | Maximum accepted output resolution |
| `ARTIFACT_STORAGE` | `s3` for external storage; `volume` for a mounted volume |
| `ALLOW_EPHEMERAL_CACHE` | Explicitly allow no model-cache volume when set to `1` |
| `HF_HOME` | Writable Hugging Face cache path |
| `TORCH_HOME` | Writable Torch cache path |
| `XDG_CACHE_HOME` | Writable helper cache path |
| `VOLUME_ROOT` | Mount root if volume storage is used |
| `PIXAL_MODEL_CACHE_HUB` | Optional read-only RunPod Hugging Face hub cache root |

RunPod's optional [cached model](https://docs.runpod.io/serverless/endpoints/model-caching)
feature works without a network volume. Choose the Pixal3D repository in the
console's Cached model field and set `PIXAL_MODEL_CACHE_HUB` to the platform's
Hugging Face hub cache root. The worker checks for the exact pinned revision and
all required multi-view files. If they exist, it reads them in place and writes
its config overlay to the writable helper cache. Otherwise it falls back to a
selective download. Helper models still use the writable cache configured above.

RunPod currently caches an entire selected repository, so its host cache can
contain the upstream single-view weights as well. This service loads only the
multi-view models. Those cached model files are separate from external job
artifacts and do not require a paid network volume.

Without a persistent cache, each fresh worker downloads the roughly 26–30 GB
model cache. This increases cold start time and billed startup work. Temporary
files disappear when the worker is terminated; external input/output objects
remain in S3. Each warm worker initializes one backend before accepting jobs and
serializes requests. Queue the whole batch to keep it warm. The model is not
reloaded between ordinary jobs.

Without explicit `ALLOW_EPHEMERAL_CACHE=1` in external S3 mode, the worker requires
a model-cache volume. Adding one later is optional. RunPod can disable max workers
after seven days of endpoint inactivity; restore max workers in the console if
an old endpoint leaves a new batch queued.

## Client

```sh
python -m pip install -r tools/image_to_3d/runpod_service/requirements-client.txt
```

Provide credentials through runtime environment variables or a secret manager,
never in tracked files. RunPod API keys are client-only. External S3 credentials
are needed on both the client and the GPU worker.

| Variable | Purpose |
| --- | --- |
| `RUNPOD_ENDPOINT_ID` | Queue endpoint ID |
| `RUNPOD_API_KEY` | Restricted endpoint key; alternatively use the hidden prompt |
| `S3_BUCKET` | Your external bucket name |
| `S3_ENDPOINT_URL` | Your provider's HTTPS S3 endpoint; omit for AWS S3 |
| `S3_REGION` | Provider region, default `us-east-1` |
| `S3_ACCESS_KEY_ID` | Bucket-scoped access key ID |
| `S3_SECRET_ACCESS_KEY` | Bucket-scoped secret access key |
| `S3_SESSION_TOKEN` | Optional temporary-credential token |
| `S3_ADDRESSING_STYLE` | `path` (default), `virtual` or `auto`, as required by provider |

Set **`ARTIFACT_STORAGE=s3` on the worker** to use an external S3-compatible
provider for all input images and GLBs. The client uploads inputs directly to
your bucket; the worker downloads them, generates the model, uploads the GLB,
then writes completion metadata last. Clients download directly from the bucket
without needing a running GPU. The bucket stays private. Grant workers read
access to `pixal3d/inputs/` and read/write access to `pixal3d/outputs/`; clients
need input write and output read. Neither needs bucket administration permissions.

For Hetzner Object Storage, set the bucket's location endpoint and region in the
runtime environment and use virtual-hosted addressing. Bucket selection is
independent of the chosen EU RunPod GPU datacenter. Transfers from Hetzner to
RunPod count as external egress even within Europe. See the
[Hetzner S3 setup guide](https://docs.hetzner.com/storage/object-storage/getting-started/using-s3-api-tools/).

The S3 client also supports the normal AWS credential chain if explicit `S3_*`
keys are absent. Do not embed secrets in endpoint URLs or request payloads.
A private MinIO deployment must be reachable over HTTPS from RunPod workers and
the client. Provider request and data-transfer charges can apply.

Model caching is independent of artifact storage. An optional network volume
for the roughly 26–30 GB model cache avoids repeated checkpoint downloads. To
use **no RunPod network volume**, set `ALLOW_EPHEMERAL_CACHE=1` alongside
`ARTIFACT_STORAGE=s3`, set `HF_HOME`, `TORCH_HOME` and `XDG_CACHE_HOME` to writable
container-disk cache paths, and allocate enough container disk for the image,
cache and temporary job files. Each fresh worker then downloads weights again;
that can substantially increase startup time and GPU cost.

Legacy RunPod-volume storage remains available with `ARTIFACT_STORAGE=volume`.
In that mode, use `RUNPOD_VOLUME_ID`, `RUNPOD_S3_ENDPOINT`, `RUNPOD_S3_REGION`,
`RUNPOD_S3_ACCESS_KEY_ID` and `RUNPOD_S3_SECRET_ACCESS_KEY` on the client instead
of the generic S3 variables. The worker accesses its mounted volume directly and
needs no storage keys. Generic S3 variables take precedence if both are set.

Generate and download in one command:

```sh
python tools/image_to_3d/runpod_service/client.py generate \
  front.png right.png back.png left.png \
  --cameras transforms.json -o output/vehicle.glb
```

This uploads the original image bytes, submits the job, polls for completion,
and downloads the verified GLB. A local `.job.json` ticket beside the output
allows the same command to resume without generating a duplicate model. A
matching ticket must have the same image checksums, camera metadata and options.
If an output already exists, use `--overwrite` to download it again.

For a prepared batch manifest:

```sh
python tools/image_to_3d/runpod_service/client.py batch \
  --manifest vehicles.json --output-dir output/vehicles
```

A manifest contains an `assets` array. Each entry has an `id`, an `images` array
of `{ "path": "...", "sha256": "..." }` records, and a `cameras` JSON file path.
Paths are relative to the current working directory, or absolute. Use
`--asset ID` (repeatable) to select a subset. The client checks every selected
asset before submitting any jobs, queues the batch first to keep a worker warm,
then downloads each result. Re-running reuses matching saved tickets; add
`--overwrite` if some local outputs already exist.

All connection configuration comes from runtime environment variables. The
local client reads no Kubernetes Secret or deployment settings file. RunPod
Secret references configured on a worker are independent of the local client's
S3 environment. The hidden API-key prompt is optional when the key is absent
from the environment; it is reused in memory for that client process.

Separate queue/download commands remain available:

```sh
python tools/image_to_3d/runpod_service/client.py submit \
  front.png right.png back.png left.png \
  --cameras transforms.json --ticket jobs/vehicle.json

python tools/image_to_3d/runpod_service/client.py fetch jobs/vehicle.json \
  -o output/vehicle.glb
```

Options include `--faces 100000`, `--texture-size 2048`, `--steps 12`, `--seed 42`
and `--resolution 1024`. Higher resolution needs explicit server configuration
and memory testing. Use a new ticket file for each submission. Submission is not
automatically retried because a lost HTTP response can still represent an
already-created job.

Images go to `pixal3d/inputs/<upload-id>/`; small queue payloads reference those
keys. Successful outputs are committed to `pixal3d/outputs/multi/<job-id>/` with
checksums. Completed job retries reuse committed output; failures do not publish
partial GLBs. Tickets contain identifiers and input provenance, not credentials.

`fetch` checks persistent storage first, so completed jobs remain downloadable
**without a GPU worker or RunPod status API call**, including after the async API's
30-minute result retention. Downloads are checksum- and GLB-validated. Use
`--overwrite` to replace an existing output. Inputs and outputs remain on the
volume until explicitly deleted; monitor free space.

## Validation and licensing

```sh
python -m unittest discover -s tools/image_to_3d/runpod_service -p 'test_*.py' -v
```

Sixteen CPU tests cover initialization reuse, retry reuse, conflicting job IDs,
partial-output failure handling, camera/path validation, multi-only cache
selection, immutable snapshot overlays, external S3 commit/retry/failure handling and durable
download without GPU API use.
These do not verify CUDA compatibility, model quality or peak VRAM. Before normal
traffic, run two real jobs on one worker, validate the outputs, then verify worker
count reaches zero and downloads still work.

The wrapper is MIT licensed. Third-party sources and models retain their own
terms. Pixal3D is MIT; its DINOv3 dependency has separate terms. The runtime overlay
uses MIT-licensed `ZhengPeng7/BiRefNet` instead of upstream's default BRIA RMBG-2.0
background model. This does not remove other dependencies' terms.

References: [Pixal3D](https://github.com/TencentARC/Pixal3D),
[RunPod queue API](https://docs.runpod.io/serverless/endpoints/send-requests),
[endpoint settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations),
[restricted API keys](https://docs.runpod.io/get-started/credentials),
[S3 API](https://docs.runpod.io/storage/s3-api).
