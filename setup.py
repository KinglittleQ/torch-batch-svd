from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
from pathlib import Path


LIB_NAME = "torch_batch_svd"
MAJOR = 1
MINOR = 0
PATCH = 0

root_dir = Path(LIB_NAME)
include_dir = root_dir / 'include'
ext_src = [str(x.absolute()) for x in root_dir.glob('csrc/*.cpp')]
print(ext_src)

cuda_extension = CUDAExtension(
    LIB_NAME + '._c',
    sources=ext_src,
    libraries=["cusolver", "cublas"],
    extra_compile_args={'cxx': ['-O2', '-I{}'.format(str(include_dir))],
                        'nvcc': ['-O2']}
)

setup(
    name=LIB_NAME,
    version='{}.{}.{}'.format(MAJOR, MINOR, PATCH),
    packages=find_packages(exclude=('tests', 'build', 'csrc',
                                    'include', 'torch_batch_svd.egg-info')),
    ext_modules=[cuda_extension],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
)
