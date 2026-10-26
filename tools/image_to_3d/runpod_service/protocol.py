"""Small, dependency-free job contract shared by the client and GPU worker."""
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

MODEL_REVISION = 'b0cb2e1b794cab9aa0ac38a95d794a4d9337437f'
SOURCE_REVISION = 'f7cf38429b0bd264f1995f0f8743a88b1c728b94'
PREFIX = 'pixal3d'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer between {low} and {high}.')
    return value


def validate_cameras(meta, count):
    if not isinstance(meta, dict) or not isinstance(meta.get('frames'), list) or len(meta['frames']) != count:
        raise ValueError('cameras.frames must contain one camera per image, in input order.')
    for frame in meta['frames']:
        if not isinstance(frame, dict):
            raise ValueError('Each camera frame must be an object.')
        matrix = frame.get('transform_matrix')
        if not isinstance(matrix, list) or len(matrix) != 4 or any(
                not isinstance(row, list) or len(row) != 4 for row in matrix):
            raise ValueError('Camera transforms must be 4 by 4 matrices.')
        if any(type(v) not in (int, float) or not math.isfinite(v) for row in matrix for v in row):
            raise ValueError('Camera matrix values must be finite numbers.')
        if any(abs(a-b) > 1e-5 for a, b in zip(matrix[3], (0, 0, 0, 1))):
            raise ValueError('Camera matrices must have homogeneous last row [0,0,0,1].')
        fov = frame.get('camera_angle_x', meta.get('camera_angle_x'))
        if type(fov) not in (int, float) or not 0 < fov < math.pi:
            raise ValueError('Each camera needs camera_angle_x in radians, between 0 and pi.')
    scale = meta.get('mesh_scale', 1)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('mesh_scale must be positive and finite.')


def validate_input(data, mode, max_views=8, max_resolution=1024):
    if mode != 'multi':
        raise ValueError('This deployment supports multi-view only.')
    if not isinstance(data, dict) or data.get('mode', mode) != mode:
        raise ValueError('Request mode must be multi.')
    images = data.get('images')
    minimum, maximum = 2, max_views
    if not isinstance(images, list) or not minimum <= len(images) <= maximum:
        raise ValueError(f'{mode} endpoint requires {minimum} to {maximum} image keys.')
    for image in images:
        if not isinstance(image, str):
            raise ValueError('Image keys must be strings.')
        path = PurePosixPath(image)
        if path.is_absolute() or '..' in path.parts or path.parts[:2] != (PREFIX, 'inputs'):
            raise ValueError('Images must be keys below pixal3d/inputs/.')
        if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
            raise ValueError('Images must be PNG, JPEG or WebP files.')
    validate_cameras(data.get('cameras'), len(images))
    integer(data.get('resolution', 1024), 'resolution', 1024, max_resolution)
    if data.get('resolution', 1024) not in (1024, 1536):
        raise ValueError('resolution must be 1024 or 1536.')
    integer(data.get('seed', 42), 'seed', 0, 2**32-1)
    integer(data.get('steps', 12), 'steps', 4, 50)
    integer(data.get('faces', 100000), 'faces', 1000, 1000000)
    if data.get('texture_size', 2048) not in (1024, 2048, 4096):
        raise ValueError('texture_size must be 1024, 2048 or 4096.')
    if 'fov_radians' in data:
        raise ValueError('Supply per-view camera FOVs in cameras metadata.')


def job_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}', value):
        raise ValueError('Invalid job ID.')
    return value


def fingerprint(data, mode):
    return hashlib.sha256(json.dumps({'mode': mode, 'input': data, 'model_revision': MODEL_REVISION},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()
