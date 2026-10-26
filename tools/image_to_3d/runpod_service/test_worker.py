import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import glb

import handler
import client
from protocol import validate_input
from model_cache import required_files, prepare_model_path, model_snapshot


def glb_bytes():
    doc = json.dumps({'asset': {'version': '2.0'}, 'buffers': [{'byteLength': 36}],
        'bufferViews': [{'buffer': 0, 'byteLength': 36}],
        'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': 3, 'type': 'VEC3',
                      'min': [0,0,0], 'max': [1,1,0]}],
        'meshes': [{'primitives': [{'attributes': {'POSITION': 0}}]}]}).encode()
    doc += b' ' * (-len(doc) % 4)
    return (struct.pack('<4sII', b'glTF', 2, 64+len(doc)) + struct.pack('<I4s', len(doc), b'JSON') + doc
            + struct.pack('<I4s', 36, b'BIN\x00') + struct.pack('<9f', 0,0,0,1,0,0,0,1,0))


class FakeBackend:
    def __init__(self, fail=False):
        self.calls = self.cleaned = 0
        self.fail = fail

    def generate(self, images, output, workdir, data):
        self.calls += 1
        output.write_bytes(b'partial' if self.fail else glb_bytes())
        if self.fail:
            raise RuntimeError('Simulated GPU failure')

    def cleanup(self):
        self.cleaned += 1


class MemoryS3:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self):
        self.objects, self.metadata, self.commits = {}, {}, []
        self.fail_upload = False

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()
        return {'Body': io.BytesIO(self.objects[Key]), 'ContentLength': len(self.objects[Key])}

    def head_object(self, Bucket, Key):
        return {'ContentLength': len(self.objects[Key]), 'Metadata': self.metadata.get(Key, {})}

    def upload_file(self, filename, bucket, key, ExtraArgs):
        if self.fail_upload:
            raise RuntimeError('Simulated storage failure')
        self.objects[key] = Path(filename).read_bytes()
        self.metadata[key] = ExtraArgs['Metadata']
        self.commits.append(key)

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body
        self.commits.append(Key)


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        inputs = self.root/'pixal3d/inputs/run'
        inputs.mkdir(parents=True)
        for name in ('front.png', 'rear.png'):
            (inputs/name).write_bytes(b'fixture image')
        matrix = [[1,0,0,0], [0,0,-1,-3], [0,1,0,0], [0,0,0,1]]
        self.data = {'mode': 'multi', 'images': ['pixal3d/inputs/run/front.png', 'pixal3d/inputs/run/rear.png'],
            'cameras': {'camera_angle_x': .35, 'frames': [{'transform_matrix': matrix}, {'transform_matrix': matrix}]}}

    def test_model_cache_selects_multiview_files_only(self):
        config = {'args': {'models': {'flow': 'ckpts/flow_mv', 'decoder': 'ckpts/decoder'},
                           'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}}}}
        files = required_files(config)
        self.assertIn('ckpts/flow_mv.safetensors', files)
        self.assertNotIn('pipeline.json', files)
        self.assertNotIn('ckpts/flow.safetensors', files)
        with self.assertRaises(ValueError):
            required_files({'args': {'models': {'flow': 'microsoft/TRELLIS.2-4B'}}})

    def test_model_overlay_preserves_snapshot_and_avoids_bria(self):
        snapshot = self.root/'snapshot'
        (snapshot/'ckpts').mkdir(parents=True)
        config = {'args': {'models': {'flow': 'ckpts/flow_mv'},
                           'rembg_model': {'args': {'model_name': 'briaai/RMBG-2.0'}}}}
        original = json.dumps(config)
        (snapshot/'pipeline_mv.json').write_text(original)
        for extension in ('.json', '.safetensors'):
            (snapshot/('ckpts/flow_mv'+extension)).write_bytes(b'fixture')
        selected = []
        def download(*args, **kwargs):
            selected.extend(kwargs['allow_patterns'])
            return str(snapshot)
        hub = types.SimpleNamespace(hf_hub_download=lambda *args, **kwargs: str(snapshot/'pipeline_mv.json'),
                                    snapshot_download=download)
        with patch.dict(sys.modules, {'huggingface_hub': hub}), \
             patch.dict(os.environ, {'HF_HOME': str(self.root/'cache/huggingface')}):
            overlay = Path(prepare_model_path())
        self.assertEqual((snapshot/'pipeline_mv.json').read_text(), original)
        self.assertEqual(json.loads((overlay/'pipeline_mv.json').read_text())['args']['rembg_model']['args']['model_name'],
                         'ZhengPeng7/BiRefNet')
        self.assertNotIn('pipeline.json', selected)

    def test_read_only_platform_cache_avoids_hub_download(self):
        from protocol import MODEL_REVISION
        hub = self.root/'platform-hub'
        snapshot = hub/'models--TencentARC--Pixal3D'/'snapshots'/MODEL_REVISION
        (snapshot/'ckpts').mkdir(parents=True)
        config = {'args': {'models': {'flow': 'ckpts/flow_mv'}}}
        (snapshot/'pipeline_mv.json').write_text(json.dumps(config))
        for extension in ('.json', '.safetensors'):
            (snapshot/('ckpts/flow_mv'+extension)).write_bytes(b'fixture')
        forbidden_hub = types.SimpleNamespace(hf_hub_download=lambda *a, **k: self.fail('Unexpected download'),
                                             snapshot_download=lambda *a, **k: self.fail('Unexpected download'))
        for lowercase in (False, True):
            with self.subTest(lowercase=lowercase):
                if lowercase:
                    renamed = hub/'models--tencentarc--pixal3d'
                    (hub/'models--TencentARC--Pixal3D').rename(renamed)
                    snapshot = renamed/'snapshots'/MODEL_REVISION
                with patch.dict(sys.modules, {'huggingface_hub': forbidden_hub}), \
                     patch.dict(os.environ, {'PIXAL_MODEL_CACHE_HUB': str(hub)}):
                    actual, actual_config = model_snapshot()
                self.assertEqual(actual, snapshot)
                self.assertEqual(actual_config, config)

    def test_startup_loads_once_and_serves_two_jobs(self):
        loads, results = [], []
        backend = FakeBackend()
        def construct(mode, **kwargs):
            loads.append(mode)
            return backend
        def serve(config):
            for identifier in ('job-one', 'job-two'):
                results.append(config['handler']({'id': identifier, 'input': self.data}))
        with patch.dict(sys.modules, {'runpod': types.SimpleNamespace(serverless=types.SimpleNamespace(start=serve)),
                                      'backend': types.SimpleNamespace(PixalBackend=construct)}), \
             patch.dict(os.environ, {'VOLUME_ROOT': str(self.root), 'PIXAL_MODE': 'multi'}), \
             patch.object(Path, 'is_mount', return_value=True):
            handler.main()
        self.assertEqual(loads, ['multi'])
        self.assertEqual(backend.calls, 2)
        self.assertEqual(len(results), 2)

    def test_retry_reuses_committed_output(self):
        backend = FakeBackend()
        worker = handler.Worker(backend, self.root)
        event = {'id': 'job-one', 'input': self.data}
        first, second = worker(event), worker(event)
        self.assertEqual(first, second)
        self.assertEqual(backend.calls, 1)
        glb.validate_glb(self.root/first['artifact_key'])

    def test_reused_id_with_different_request_is_rejected(self):
        worker = handler.Worker(FakeBackend(), self.root)
        worker({'id': 'job-one', 'input': self.data})
        with self.assertRaisesRegex(ValueError, 'different request'):
            worker({'id': 'job-one', 'input': {**self.data, 'seed': 123}})

    def test_failure_never_publishes_partial_output_and_cleans_up(self):
        backend = FakeBackend(fail=True)
        with self.assertRaises(RuntimeError):
            handler.Worker(backend, self.root)({'id': 'job-one', 'input': self.data})
        self.assertFalse(list(self.root.glob('pixal3d/outputs/**/*.glb')))
        self.assertEqual(backend.cleaned, 1)

    def test_path_traversal_and_wrong_camera_count_are_rejected(self):
        for data in ({**self.data, 'images': ['pixal3d/inputs/../../etc/a.png']*2},
                     {**self.data, 'cameras': {'frames': []}}):
            with self.assertRaises(ValueError):
                validate_input(data, 'multi')

    def test_symlink_escape_is_rejected(self):
        external = self.root/'outside.png'
        external.write_bytes(b'outside')
        link = self.root/'pixal3d/inputs/run/escape.png'
        link.symlink_to(external)
        data = {**self.data, 'images': ['pixal3d/inputs/run/escape.png']*2}
        with self.assertRaisesRegex(ValueError, 'permitted'):
            handler.Worker(FakeBackend(), self.root)({'id': 'job-one', 'input': data})

    def external_worker(self, fail_upload=False):
        from s3_worker import S3Worker
        storage = MemoryS3()
        for key in self.data['images']:
            storage.objects[key] = b'fixture image'
        storage.fail_upload = fail_upload
        backend = FakeBackend()
        return S3Worker(backend, storage, 'external-bucket'), backend, storage

    def test_external_s3_commits_metadata_last_and_reuses_retry(self):
        worker, backend, storage = self.external_worker()
        event = {'id': 'external-one', 'input': self.data}
        first = worker(event)
        self.assertEqual(worker(event), first)
        self.assertEqual(backend.calls, 1)
        self.assertTrue(storage.commits[-1].endswith('result.json'))
        self.assertTrue(storage.commits[-2].endswith('model.glb'))
        with self.assertRaisesRegex(ValueError, 'different request'):
            worker({**event, 'input': {**self.data, 'seed': 123}})

    def test_external_s3_failed_upload_does_not_commit_result(self):
        worker, backend, storage = self.external_worker(fail_upload=True)
        with self.assertRaises(RuntimeError):
            worker({'id': 'external-one', 'input': self.data})
        self.assertFalse(storage.commits)
        self.assertEqual(backend.cleaned, 1)

    def test_external_s3_rejects_oversized_stream_before_inference(self):
        from s3_worker import MAX_IMAGE_BYTES
        worker, backend, storage = self.external_worker()
        original = storage.get_object
        def oversized(**kwargs):
            result = original(**kwargs)
            if kwargs['Key'] in self.data['images']:
                result['ContentLength'] = MAX_IMAGE_BYTES+1
            return result
        storage.get_object = oversized
        with self.assertRaisesRegex(ValueError, '32 MiB'):
            worker({'id': 'external-one', 'input': self.data})
        self.assertEqual(backend.calls, 0)

    def client_args(self):
        cameras = self.root/'cameras.json'
        cameras.write_text(json.dumps(self.data['cameras']))
        return argparse.Namespace(images=[self.root/key for key in self.data['images']], cameras=cameras,
            output=self.root/'output.glb', ticket=None, endpoint='endpoint', mode='multi',
            resolution=1024, seed=42, steps=12, faces=100000, texture_size=2048,
            timeout=1, overwrite=False)

    def test_generate_resumes_matching_ticket_without_resubmitting(self):
        args = self.client_args()
        images, _, data = client.input_request(args, 'original-upload')
        ticket = args.output.with_suffix('.job.json')
        ticket.write_text(json.dumps({'input': data, 'input_sha256': [client.file_hash(p) for p in images],
                                     'endpoint_id': 'endpoint'}))
        with patch.object(client, 'submit', side_effect=AssertionError('Must reuse the existing job')), \
             patch.object(client, 'fetch') as fetch:
            client.generate(args)
        fetch.assert_called_once()
        args.seed += 1
        with self.assertRaisesRegex(ValueError, 'different inputs'):
            client.generate(args)

    def batch_args(self):
        args = self.client_args()
        records = [{'path': str(p), 'sha256': client.file_hash(p)} for p in args.images]
        args.manifest = self.root/'batch.json'
        args.manifest.write_text(json.dumps({'assets': [
            {'id': ident, 'images': records, 'cameras': str(args.cameras)} for ident in ('V01', 'V03')]}))
        args.output_dir, args.asset = self.root/'models', None
        return args

    def test_batch_queues_every_job_before_waiting_for_results(self):
        args, events = self.batch_args(), []
        with patch.object(client, 'submit', side_effect=lambda a: events.append(('submit', a.output.parent.name))), \
             patch.object(client, 'fetch', side_effect=lambda a: events.append(('fetch', a.output.parent.name))):
            client.batch(args)
        self.assertEqual(events, [('submit', 'V01'), ('submit', 'V03'), ('fetch', 'V01'), ('fetch', 'V03')])

    def test_batch_preflights_all_assets_before_any_billed_submission(self):
        args = self.batch_args()
        doc = json.loads(args.manifest.read_text())
        doc['assets'][1]['images'][0]['sha256'] = 'changed'
        args.manifest.write_text(json.dumps(doc))
        with patch.object(client, 'submit') as submit:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                client.batch(args)
        submit.assert_not_called()

    def test_completed_result_download_does_not_need_runpod_api(self):
        result = handler.Worker(FakeBackend(), self.root)({'id': 'job-one', 'input': self.data})
        class Storage:
            def get_object(self, **kwargs):
                return {'Body': io.BytesIO(json.dumps(result).encode())}
            def download_file(self, bucket, key, filename):
                Path(filename).write_bytes(glb_bytes())
        ticket = self.root/'ticket.json'
        ticket.write_text(json.dumps({'job_id': 'job-one', 'volume_id': 'volume', 'endpoint_id': 'endpoint',
                                     'metadata_key': 'pixal3d/outputs/multi/job-one/result.json'}))
        output = self.root/'download.glb'
        with patch.object(client, 's3_client', return_value=Storage()), \
             patch.object(client, 'RunpodAPI', side_effect=AssertionError('No API or GPU wake-up needed')):
            client.fetch(argparse.Namespace(ticket=ticket, output=output, overwrite=False, timeout=1))
        glb.validate_glb(output)


if __name__ == '__main__':
    unittest.main()
