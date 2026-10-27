"""CPU-only proof of native kernel coverage; does not initialize CUDA."""
from pathlib import Path
import glob,re,site,subprocess

patterns={
    'CuMesh':'cumesh/_C*.so',
    'CuMesh BVH':'cumesh/_cubvh*.so',
    'FlexGEMM':'flex_gemm/kernels/cuda*.so',
    'o-voxel':'o_voxel/_C*.so',
}
for name,pattern in patterns.items():
    matches=[p for root in site.getsitepackages() for p in glob.glob(str(Path(root)/pattern))]
    if len(matches)!=1:raise RuntimeError('Expected exactly one '+name+' binary')
    output=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--list-elf',matches[0]],text=True)
    arches=set(re.findall(r'sm_(\d+)',output))
    missing={'80','89','90'}-arches
    if missing:raise RuntimeError(name+' lacks native architectures: '+','.join(sorted(missing)))
    print(name+': sm_80, sm_89, sm_90 verified')
