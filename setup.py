from setuptools import find_packages
from setuptools import setup
import sys
from sultan import Sultan
import pathlib
import os
import torch
from torch.utils.ffi import create_extension

CUDA_ARCH = ('-gencode arch=compute_30,code=sm_30 '
             '-gencode arch=compute_35,code=sm_35 '
             '-gencode arch=compute_50,code=sm_50 '
             '-gencode arch=compute_52,code=sm_52 '
             '-gencode arch=compute_60,code=sm_60 '
             '-gencode arch=compute_61,code=sm_61')

project_dir = pathlib.Path(__file__).resolve().parent
finished_file = project_dir / 'finished_install.txt'
src = project_dir / 'retinanet/lib/nms/src'

if not finished_file.exists():
    with Sultan.load(cwd=project_dir / 'retinanet/lib/nms/src/cuda') as s:
        cmd = ('/usr/local/cuda/bin/nvcc -c -o '
               'nms_kernel.cu.o nms_kernel.cu -x '
               'cu -Xcompiler -fPIC ' + CUDA_ARCH)
        s.bash(f'-c "{cmd}"').run()

    sources = [src / 'nms.c']
    headers = [src / 'nms.h']
    defines = []
    with_cuda = False

    if torch.cuda.is_available():
        print('Including CUDA code.')
        sources += [src / 'nms_cuda.c']
        headers += [src / 'nms_cuda.h']
        defines += [('WITH_CUDA', None)]
        with_cuda = True

    extra_objects = [src / 'cuda/nms_kernel.cu.o']
    ffi = create_extension('retinanet.lib.nms._ext.nms',
                           headers=[str(x) for x in headers],
                           sources=[str(x) for x in sources],
                           define_macros=defines,
                           relative_to=__file__,
                           with_cuda=with_cuda,
                           extra_objects=[str(x) for x in extra_objects],
                           extra_compile_args=['-std=c99'])
    ffi.build()
    finished_file.touch()

setup(name='retinanet',
      packages=find_packages())
