from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os


# export CUDA_HOME=<YOUR_CUDA_DIR>
conda = os.getenv("CUDA_HOME")
if conda:
    inc = [conda + "/include"]
else:
    inc = []

libname = "torch_batch_svd"
setup(name=libname,
      ext_modules=[CppExtension(
          libname,
          [libname + '.cpp'],
          include_dirs=inc,
          libraries=["cusolver", "cublas"],
          extra_compile_args={'cxx': ['-g', '-DDEBUG'],
                              'nvcc': ['-O2']}
      )],
      cmdclass={'build_ext': BuildExtension})
