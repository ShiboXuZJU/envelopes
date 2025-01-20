from setuptools import setup, Extension
import pybind11

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

setup(packages=['envelopes'],
      ext_modules=[envelopes],
      include_package_data=True,
      package_data={"envelopes": ["*.dll", "*.pyi"]})
