from setuptools import setup, Extension
import pybind11

__version__ = '0.0.1'

envelopes = Extension('envelopes.envelopes_cpp',
                      sources=['src/bind.cpp'],
                      include_dirs=[pybind11.get_include()],
                      language='c++',
                      depends=[
                          'src/envelopes.cpp', 'src/envelopes.h',
                          'src/libcerf/cerf.h', 'src/libcerf/cerfcpp.lib'
                      ],
                      libraries=['src/libcerf/cerfcpp'],
                      extra_compile_args=['/std:c++17'])

setup(name='envelopes',
      author="Shibo Xu",
      author_email="sbxu@zju.edu.cn",
      version=__version__,
      description="Envelops impelemented both in C++ and python.",
      requires=["numpy", "scipy"],
      packages=['envelopes'],
      ext_modules=[envelopes],
      include_package_data=True,
      package_data={"envelopes": ["*.dll", "*.pyi"]})
