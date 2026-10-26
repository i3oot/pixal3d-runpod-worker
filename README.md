# Pixal3D multi-view on RunPod Serverless

Queue-based worker with RunPod API-key authentication, one initialized model per
warm worker, persistent textured GLB outputs and GPU scale-to-zero.
**Multi-view only: no single-view checkpoints or TRELLIS model weights.** Some
upstream TRELLIS.2 export code is required by Pixal3D and included as a dependency.

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

The worker refuses startup without the volume mount. Weights are downloaded on
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

Provide these through your local runtime environment or secret manager, never in
tracked files. No account or storage credential is required inside the worker.

| Variable | Purpose |
| --- | --- |
| `RUNPOD_ENDPOINT_ID` | Queue endpoint ID |
| `RUNPOD_API_KEY` | Restricted endpoint key; alternatively use the hidden prompt |
| `RUNPOD_VOLUME_ID` | Network-volume ID / S3 bucket name |
| `RUNPOD_S3_ENDPOINT` | Datacenter S3 API HTTPS URL |
| `RUNPOD_S3_REGION` | Matching datacenter ID |
| `RUNPOD_S3_ACCESS_KEY_ID` | Separate RunPod S3 access key ID |
| `RUNPOD_S3_SECRET_ACCESS_KEY` | Separate RunPod S3 secret access key |

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

Nine CPU tests cover initialization reuse, retry reuse, conflicting job IDs,
partial-output failure handling, camera/path validation, multi-only cache
selection, immutable snapshot overlays and durable download without GPU API use.
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
