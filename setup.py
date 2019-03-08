from setuptools import find_packages
from setuptools import setup
import sys
from sultan import Sultan
import pathlib


CUDA_ARCH = ('-gencode arch=compute_30,code=sm_30 '
             '-gencode arch=compute_35,code=sm_35 '
             '-gencode arch=compute_50,code=sm_50 '
             '-gencode arch=compute_52,code=sm_52 '
             '-gencode arch=compute_60,code=sm_60 '
             '-gencode arch=compute_61,code=sm_61')

project_dir = pathlib.Path(__file__).resolve().parent
finished_file = project_dir / 'finished_install.txt'


if not finished_file.exists():
    with Sultan.load(cwd=project_dir / 'retinanet/lib/nms/src/cuda') as s:
        cmd = ('/usr/local/cuda/bin/nvcc -c -o '
               'nms_kernel.cu.o nms_kernel.cu -x '
               'cu -Xcompiler -fPIC ' + CUDA_ARCH)
        s.bash(f'-c "{cmd}"').run()

#    with Sultan.load(cwd=project_dir / 'retinanet/lib/nms') as s:
#        s.bash(sys.executable + ' build.py').run(streaming=True)
    finished_file.touch()


setup(name='retinanet',
      packages=find_packages())
