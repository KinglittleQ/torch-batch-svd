from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import glob

libname = "torch_batch_svd"
ext_src = glob.glob(os.path.join(libname, 'csrc/*.cpp'))
print(ext_src)

setup(name=libname,
      packages=find_packages(exclude=('tests', 'build', 'csrc', 'include', 'torch_batch_svd.egg-info')),
      ext_modules=[CUDAExtension(
          libname + '._c',
          sources=ext_src,
          libraries=["cusolver", "cublas"],
          extra_compile_args={'cxx': ['-O2', '-I{}'.format('{}/include'.format(libname))],
                              'nvcc': ['-O2']}
      )],
      cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
      )
