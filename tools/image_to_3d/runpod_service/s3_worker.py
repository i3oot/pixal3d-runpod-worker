"""External object storage adapter; no durable model output depends on the GPU disk."""
import json
from pathlib import Path
import tempfile
import threading

from protocol import PREFIX, fingerprint, job_id, validate_input
from storage import read_result

MAX_IMAGE_BYTES = 32*1024*1024


class S3Worker:
    def __init__(self, backend, storage, bucket, max_views=8, max_resolution=1024):
        self.backend, self.storage, self.bucket = backend, storage, bucket
        self.max_views, self.max_resolution = max_views, max_resolution
        self.lock = threading.Lock()

    def __call__(self, job):
        from handler import Worker
        identifier = job_id(job.get('id'))
        data = job.get('input')
        validate_input(data, 'multi', self.max_views, self.max_resolution)
        metadata_key = f'{PREFIX}/outputs/multi/{identifier}/result.json'
        with self.lock:
            saved = read_result(self.storage, self.bucket, metadata_key)
            if saved is not None:
                if saved['request_hash'] != fingerprint(data, 'multi'):
                    raise ValueError('Job ID was previously used for a different request.')
                head = self.storage.head_object(Bucket=self.bucket, Key=saved['artifact_key'])
                if head['ContentLength'] != saved['bytes'] or head.get('Metadata', {}).get('sha256') != saved['sha256']:
                    raise ValueError('Stored artifact failed its size/checksum metadata check.')
                return saved
            with tempfile.TemporaryDirectory(prefix='pixal-s3-') as temporary:
                root = Path(temporary)
                for key in data['images']:
                    destination = root/key
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    response = self.storage.get_object(Bucket=self.bucket, Key=key)
                    # Bound the actual stream too, since the object can change between calls.
                    with response['Body'] as body, destination.open('wb') as output:
                        if response['ContentLength'] > MAX_IMAGE_BYTES:
                            raise ValueError('Input image exceeds 32 MiB.')
                        total = 0
                        while chunk := body.read(1024*1024):
                            total += len(chunk)
                            if total > MAX_IMAGE_BYTES:
                                raise ValueError('Input image exceeds 32 MiB.')
                            output.write(chunk)
                saved = Worker(self.backend, root, max_views=self.max_views,
                               max_resolution=self.max_resolution)(job)
                self.storage.upload_file(str(root/saved['artifact_key']), self.bucket, saved['artifact_key'],
                    ExtraArgs={'ContentType': 'model/gltf-binary', 'Metadata': {'sha256': saved['sha256']}})
                # Commit marker last: failed uploads never advertise a completed result.
                self.storage.put_object(Bucket=self.bucket, Key=metadata_key,
                    Body=json.dumps(saved).encode(), ContentType='application/json')
                return saved
