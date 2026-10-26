"""Validate self-contained GLB output before publishing or downloading."""
import json
from pathlib import Path
import struct


def validate_glb(path: Path) -> None:
    if not path.is_file():
        raise ValueError('Backend finished without creating the requested GLB.')
    with path.open('rb') as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError('Backend output is truncated.')
        magic, version, length = struct.unpack('<4sII', header)
        if magic != b'glTF' or version != 2 or length != path.stat().st_size:
            raise ValueError('Backend output is not a complete GLB 2 file.')
        chunk = stream.read(8)
        if len(chunk) != 8:
            raise ValueError('Backend output has no JSON chunk.')
        size, kind = struct.unpack('<I4s', chunk)
        if kind != b'JSON' or size > length - 20:
            raise ValueError('Backend output has an invalid JSON chunk.')
        doc = json.loads(stream.read(size))
        for resource in (*doc.get('buffers', []), *doc.get('images', [])):
            uri = resource.get('uri')
            if uri is not None and not uri.startswith('data:'):
                raise ValueError('Backend output references external resources; a self-contained GLB is required.')
        accessors = doc.get('accessors', [])
        has_geometry = False
        for mesh in doc.get('meshes', []):
            for primitive in mesh.get('primitives', []):
                position = primitive.get('attributes', {}).get('POSITION')
                if isinstance(position, int) and 0 <= position < len(accessors):
                    if accessors[position].get('count', 0) >= 3:
                        has_geometry = True
        if not has_geometry:
            raise ValueError('Backend output contains no mesh geometry.')

