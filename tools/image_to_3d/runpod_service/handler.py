"""RunPod queue handler. Results are committed to the persistent volume before success."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time

from protocol import PREFIX, MODEL_REVISION, SOURCE_REVISION, file_hash, fingerprint, job_id, validate_input


class Worker:
    def __init__(self, backend, volume, mode='multi', max_views=8, max_resolution=1024):
        self.backend, self.volume, self.mode = backend, Path(volume).resolve(), mode
        self.max_views, self.max_resolution = max_views, max_resolution
        self.lock = threading.Lock()

    def __call__(self, job):
        identifier = job_id(job.get('id'))
        data = job.get('input')
        validate_input(data, self.mode, self.max_views, self.max_resolution)
        request_hash = fingerprint(data, self.mode)
        directory = self.volume/PREFIX/'outputs'/self.mode/identifier
        with self.lock:
            metadata = directory/'result.json'
            result = directory/'model.glb'
            if metadata.exists() and result.exists():
                saved = json.loads(metadata.read_text())
                if saved['request_hash'] != request_hash:
                    raise ValueError('Job ID was previously used for a different request.')
                if saved['sha256'] == file_hash(result):
                    return saved
                raise ValueError('Stored artifact failed its checksum.')
            inputs = []
            input_root = (self.volume/PREFIX/'inputs').resolve()
            for key in data['images']:
                path = (self.volume/key).resolve()
                if not path.is_relative_to(input_root) or not path.is_file():
                    raise ValueError('Input does not exist inside the permitted input directory.')
                if path.stat().st_size > 32*1024*1024:
                    raise ValueError('Input image exceeds 32 MiB.')
                inputs.append(path)
            directory.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            try:
                with tempfile.TemporaryDirectory(prefix='.job-', dir=directory) as temporary:
                    workdir = Path(temporary)
                    pending = workdir/'model.glb'
                    self.backend.generate(inputs, pending, workdir, data)
                    from glb import validate_glb
                    validate_glb(pending)
                    saved = {'artifact_key': str(result.relative_to(self.volume)), 'sha256': file_hash(pending),
                             'bytes': pending.stat().st_size, 'mode': self.mode, 'request_hash': request_hash,
                             'model_revision': MODEL_REVISION, 'source_revision': SOURCE_REVISION,
                             'elapsed_seconds': round(time.monotonic()-started, 2)}
                    pending.replace(result)
                    pending_metadata = workdir/'result.json'
                    pending_metadata.write_text(json.dumps(saved, indent=2))
                    pending_metadata.replace(metadata)
                return saved
            finally:
                self.backend.cleanup()


def main():
    import runpod
    from backend import PixalBackend
    volume = Path(os.environ.get('VOLUME_ROOT', '/runpod-volume'))
    if not volume.is_mount():
        raise RuntimeError('Attach a persistent RunPod network volume at VOLUME_ROOT before starting this worker.')
    mode = os.environ.get('PIXAL_MODE', 'multi')
    if mode != 'multi':
        raise ValueError('This deployment supports PIXAL_MODE=multi only.')
    # This executes once at worker startup, before RunPod starts dispatching jobs.
    backend = PixalBackend(mode, low_vram=os.environ.get('PIXAL_LOW_VRAM') == '1')
    worker = Worker(backend, volume, mode, max_views=int(os.environ.get('MAX_VIEWS', '8')),
                    max_resolution=int(os.environ.get('MAX_RESOLUTION', '1024')))
    def handle_job(job):
        return worker(job)
    runpod.serverless.start({'handler': handle_job})


if __name__ == '__main__':
    main()
