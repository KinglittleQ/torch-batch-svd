from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
from pathlib import Path

libname = "torch_batch_svd"

root_dir = Path(libname)
include_dir = root_dir / 'include'
ext_src = [str(x.absolute()) for x in root_dir.glob('csrc/*.cpp')]
print(ext_src)

cuda_extension = CUDAExtension(
    libname + '._c',
    sources=ext_src,
    libraries=["cusolver", "cublas"],
    extra_compile_args={'cxx': ['-O2', '-I{}'.format(str(include_dir))],
                        'nvcc': ['-O2']}
)

setup(
    name=libname,
    packages=find_packages(exclude=('tests', 'build', 'csrc',
                                    'include', 'torch_batch_svd.egg-info')),
    ext_modules=[cuda_extension],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
)
