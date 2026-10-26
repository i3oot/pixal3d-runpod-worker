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
first startup is measured. A **50 GB network volume** leaves initial room for
inputs and outputs. The full upstream repository is about 46 GB because it also
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

`endpoint-settings.json` is a console checklist, not an API request body:

- Queue-based endpoint, one GPU per worker; active workers 0, max workers 1.
- Idle timeout 60 seconds, execution timeout 1800 seconds, job TTL 24 hours.
- Attach the persistent 50 GB network volume at `/runpod-volume`; choose a region
  with compatible GPU capacity and RunPod's S3 API.
- Set `PIXAL_MODE=multi`, `PIXAL_LOW_VRAM=1`, `MAX_VIEWS=4`, `MAX_RESOLUTION=1024`.
- Use a restricted RunPod API key scoped to this endpoint on the client.

By default the worker requires a model-cache volume. Weights are downloaded on
first startup and reused from persistent storage on later cold starts. Each warm
worker initializes one backend before accepting jobs and serializes requests.
Queue the whole batch to keep one worker warm. After the queue drains, the idle
timeout lets RunPod terminate it; the next cold worker must initialize again.
RunPod can disable max workers after seven days of endpoint inactivity; restore
max workers to one if an old endpoint leaves a new batch queued.

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

For Hetzner Object Storage, use the bucket's location endpoint, for example
`S3_ENDPOINT_URL=https://fsn1.your-objectstorage.com`, `S3_REGION=fsn1` and
`S3_ADDRESSING_STYLE=virtual` (or the matching `nbg1`/`hel1` values). Endpoint
configuration is separate from the chosen EU RunPod GPU datacenter. Transfer
from Hetzner to RunPod is external egress, even when both are in Europe; it uses
the Hetzner account's traffic allowance and may incur overage charges. See the
[Hetzner S3 setup guide](https://docs.hetzner.com/storage/object-storage/getting-started/using-s3-api-tools/).

The S3 client also supports the normal AWS credential chain if explicit `S3_*`
keys are absent. Do not embed secrets in endpoint URLs or request payloads.
A private MinIO deployment must be reachable over HTTPS from RunPod workers and
the client. Provider request and data-transfer charges can apply.

Model caching is independent of artifact storage. Keeping the network volume
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

Twelve CPU tests cover initialization reuse, retry reuse, conflicting job IDs,
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
