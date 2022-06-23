from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='num_sys',
      ext_modules=[cpp_extension.CppExtension('num_sys', ['num_sys.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

