#!/usr/bin/env python

from distutils.core import setup

setup(name='bird_utils',
    version='0.0',
    description='Random utilities I wrote',
    author='bird',
    author_email='jjbird@gmail.com',
    package_dir = {'': 'src'},
    packages=['geodesy', 'geometry', 'meteorology', 'parsers'],)
