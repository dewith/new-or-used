# numpydoc ignore=SS03
"""
A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()
_readme = (here / 'README.md').read_text(encoding='utf-8')
_license = (here / 'LICENSE').read_text(encoding='utf-8')


setup(
    name='new-or-used',
    version='0.1.0',
    description='Item Condition Classification',
    long_description=_readme,
    url='https://github.com/dewith/new-or-used/',
    author='Dewith Miramón',
    author_email='dewithmiramon@gmail.com',
    license=_license,
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)
