import subprocess

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from pathlib import Path

LIB_NAME = "torch_batch_svd"

rev = "+" + subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").rstrip()
version = "1.1.0" + rev

root_dir = Path(LIB_NAME)
include_dir = root_dir / "include"
ext_src = [str(x.absolute()) for x in root_dir.glob("csrc/*.cpp")]

cuda_extension = CUDAExtension(
    LIB_NAME + "._c",
    sources=ext_src,
    libraries=["cusolver", "cublas"],
    include_dirs=[include_dir],
    extra_compile_args={"cxx": ["-O2", "-Wno-unknown-pragmas"], "nvcc": ["-O2"]},
)

setup(
    name=LIB_NAME,
    version=version,
    description="A 100x faster PyTorch SVD",
    author="Chengqi Deng",
    license="MIT",
    python_requires=">=3.6",
    install_requires=["torch>=1.0"],
    packages=[LIB_NAME],
    ext_modules=[cuda_extension],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
