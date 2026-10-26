#!/usr/bin/env python3
"""Upload references, queue Pixal3D jobs and download durable results."""
import argparse
import getpass
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request
import uuid

from protocol import PREFIX, file_hash, job_id, validate_input


from storage import bucket_name, read_result, s3_client


class RunpodAPI:
    def __init__(self, endpoint):
        self.endpoint = job_id(endpoint)
        self.key = os.environ.get('RUNPOD_API_KEY') or getpass.getpass('RunPod API key: ')
        if not self.key:
            raise ValueError('RunPod API key is required.')

    def request(self, operation, payload=None):
        request = urllib.request.Request(f'https://api.runpod.ai/v2/{self.endpoint}/{operation}',
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={'Authorization': 'Bearer '+self.key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'RunPod returned HTTP {error.code}.') from None


def submit(args):
    if args.ticket.exists():
        raise ValueError('Ticket already exists; use a new ticket path or fetch the existing job.')
    storage, bucket = s3_client(), bucket_name()
    api = RunpodAPI(args.endpoint or os.environ['RUNPOD_ENDPOINT_ID'])
    run = uuid.uuid4().hex
    images = [path.resolve() for path in args.images]
    for image in images:
        if not image.is_file() or image.stat().st_size > 32*1024*1024:
            raise ValueError('An input image is missing or exceeds 32 MiB.')
    keys = [f'{PREFIX}/inputs/{run}/{index:02d}{path.suffix.lower()}' for index, path in enumerate(images)]
    data = {'mode': args.mode, 'images': keys, 'resolution': args.resolution, 'seed': args.seed,
            'steps': args.steps, 'faces': args.faces, 'texture_size': args.texture_size}
    if args.cameras:
        data['cameras'] = json.loads(args.cameras.read_text())
    validate_input(data, args.mode, max_resolution=1536)
    for image, key in zip(images, keys):
        storage.upload_file(str(image), bucket, key)
    submitted = api.request('run', {'input': data})
    identifier = job_id(submitted['id'])
    ticket = {'job_id': identifier, 'endpoint_id': api.endpoint, 'bucket': bucket, 'mode': args.mode,
              'metadata_key': f'{PREFIX}/outputs/{args.mode}/{identifier}/result.json',
              'input': data, 'input_sha256': [file_hash(image) for image in images]}
    args.ticket.parent.mkdir(parents=True, exist_ok=True)
    with args.ticket.open('x') as stream:
        json.dump(ticket, stream, indent=2)
    print(f'Queued {identifier}; ticket: {args.ticket}')


def download(storage, bucket, result, output, overwrite=False):
    if output.exists() and not overwrite:
        raise ValueError('Output exists; use --overwrite to replace it.')
    if output.suffix.lower() != '.glb':
        raise ValueError('Output must end in .glb.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pixal-download-', dir=output.parent) as temporary:
        pending = Path(temporary)/'model.glb'
        storage.download_file(bucket, result['artifact_key'], str(pending))
        if pending.stat().st_size != result['bytes'] or file_hash(pending) != result['sha256']:
            raise ValueError('Downloaded GLB failed size/checksum verification.')
        from glb import validate_glb
        validate_glb(pending)
        if overwrite:
            pending.replace(output)
        else:
            os.link(pending, output)
    print(f'Downloaded: {output}')


def fetch(args):
    ticket = json.loads(args.ticket.read_text())
    storage = s3_client()
    bucket = ticket.get('bucket') or ticket['volume_id']
    started, api = time.monotonic(), None
    while True:
        # Check durable storage first: no GPU wake-up and no dependency on job-result retention.
        result = read_result(storage, bucket, ticket['metadata_key'])
        if result is not None:
            download(storage, bucket, result, args.output.absolute(), args.overwrite)
            return
        if api is None:
            api = RunpodAPI(ticket['endpoint_id'])
        status = api.request('status/'+job_id(ticket['job_id']))
        state = status['status']
        if state in ('FAILED', 'CANCELLED', 'TIMED_OUT'):
            raise RuntimeError(f'Job {state}; inspect the RunPod worker logs.')
        if time.monotonic()-started > args.timeout:
            raise TimeoutError('Still pending; the job remains queued/running. Re-run fetch later with the same ticket.')
        print(f'{ticket["job_id"]}: {state}', flush=True)
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    enqueue = commands.add_parser('submit')
    enqueue.add_argument('images', type=Path, nargs='+')
    enqueue.add_argument('--ticket', type=Path, required=True)
    enqueue.add_argument('--endpoint')
    enqueue.add_argument('--mode', choices=('multi',), default='multi')
    enqueue.add_argument('--cameras', type=Path)
    enqueue.add_argument('--resolution', type=int, choices=(1024, 1536), default=1024)
    enqueue.add_argument('--seed', type=int, default=42)
    enqueue.add_argument('--steps', type=int, default=12)
    enqueue.add_argument('--faces', type=int, default=100000)
    enqueue.add_argument('--texture-size', type=int, default=2048)
    enqueue.set_defaults(action=submit)
    retrieve = commands.add_parser('fetch')
    retrieve.add_argument('ticket', type=Path)
    retrieve.add_argument('-o', '--output', type=Path, required=True)
    retrieve.add_argument('--timeout', type=float, default=10800)
    retrieve.add_argument('--overwrite', action='store_true')
    retrieve.set_defaults(action=fetch)
    args = parser.parse_args()
    from botocore.exceptions import BotoCoreError, ClientError
    try:
        args.action(args)
    except ClientError as error:
        print('Storage error: '+error.response.get('Error', {}).get('Code', 'ClientError'))
        return 1
    except BotoCoreError as error:
        print('Storage error: '+type(error).__name__)
        return 1
    except (KeyError, ValueError, OSError, RuntimeError) as error:
        print(f'Error: {error}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
