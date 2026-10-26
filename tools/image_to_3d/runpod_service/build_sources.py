"""Fetch pinned CUDA extension sources during the container build."""
import json
from pathlib import Path
import subprocess


def main():
    lock = json.loads(Path('/opt/worker/source-lock.json').read_text())
    names = {'trellis2': 'TRELLIS.2', 'cumesh': 'CuMesh', 'flexgemm': 'FlexGEMM',
             'nvdiffrec': 'nvdiffrec', 'pixal3d': 'Pixal3D'}
    for name, directory in names.items():
        item = lock[name]
        destination = Path('/opt')/directory
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        'https://github.com/'+item['repo']+'.git', str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', item['revision']], check=True)
        subprocess.run(['git', '-C', str(destination), 'submodule', 'update', '--init', '--recursive'], check=True)
    subprocess.run(['git', 'clone', '--branch', 'v0.4.0', '--depth', '1',
                    'https://github.com/NVlabs/nvdiffrast.git', '/opt/nvdiffrast'], check=True)
    # MoGe is single-view only; Gradio is an interactive UI. Neither is used by this worker.
    requirements = Path('/opt/Pixal3D/requirements.txt')
    requirements.write_text('\n'.join(line for line in requirements.read_text().splitlines()
                                    if 'microsoft/MoGe' not in line and line.strip() != 'gradio')+'\n')


if __name__ == '__main__':
    main()
