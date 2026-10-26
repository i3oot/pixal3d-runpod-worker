"""Cache only Pixal3D multi-view weights; leave the downloaded snapshot immutable."""
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

from protocol import MODEL_REVISION


def required_files(config):
    files = ['pipeline_mv.json', 'LICENSE', 'README.md']
    for name in config['args']['models'].values():
        path = PurePosixPath(name)
        if path.parts[0] != 'ckpts' or '..' in path.parts:
            raise ValueError('The pinned multi-view config must reference local Pixal3D checkpoints only.')
        files += [name+'.json', name+'.safetensors']
    return sorted(set(files))


def model_snapshot():
    # Optional read-only platform cache. Require the exact pinned revision and
    # every multi-view checkpoint; otherwise use the normal selective download.
    cached_hub = os.environ.get('PIXAL_MODEL_CACHE_HUB')
    if cached_hub:
        cached = Path(cached_hub)/'models--TencentARC--Pixal3D'/'snapshots'/MODEL_REVISION
        config_path = cached/'pipeline_mv.json'
        if config_path.is_file():
            config = json.loads(config_path.read_text())
            if all((cached/name).is_file() for name in required_files(config) if name.startswith('ckpts/')):
                return cached, config
    from huggingface_hub import hf_hub_download, snapshot_download
    path = hf_hub_download('TencentARC/Pixal3D', 'pipeline_mv.json', revision=MODEL_REVISION)
    config = json.loads(Path(path).read_text())
    files = required_files(config)
    snapshot = Path(snapshot_download('TencentARC/Pixal3D', revision=MODEL_REVISION, allow_patterns=files))
    if any(not (snapshot/name).is_file() for name in files if name.startswith('ckpts/')):
        raise RuntimeError('The Pixal3D multi-view checkpoint snapshot is incomplete.')
    return snapshot, config


def prepare_model_path():
    snapshot, config = model_snapshot()
    # Upstream's default RMBG-2.0 has separate commercial terms. Use the MIT BiRefNet
    # checkpoint with the same upstream wrapper instead; no BRIA credential is needed.
    config['args']['rembg_model']['args']['model_name'] = 'ZhengPeng7/BiRefNet'
    overlay = Path(os.environ.get('HF_HOME', '/runpod-volume/pixal3d/cache/huggingface')).parent/'pipelines'/MODEL_REVISION
    overlay.mkdir(parents=True, exist_ok=True)
    try:
        (overlay/'ckpts').symlink_to(snapshot/'ckpts', target_is_directory=True)
    except FileExistsError:
        pass
    with tempfile.NamedTemporaryFile(mode='w', dir=overlay, delete=False) as stream:
        json.dump(config, stream)
        temporary = Path(stream.name)
    temporary.replace(overlay/'pipeline_mv.json')
    return str(overlay)
