#!/usr/bin/env python

from distutils.core import setup,Command

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys,subprocess
        errno = subprocess.call([sys.executable,'runtests.py'])
        raise SystemExit(errno)

setup(name='pyica',
      version='1.0',
      description='Pure Python Package for FastICA',
      author='Kevin Brown',
      author_email='kevin.s.brown@uconn.edu',
      url='https://github.com/thelahunginjeet/pyica',
      packages=['pyica'],
      package_dir = {'pyica': ''},
      cmdclass = {'test': PyTest},
      license='BSD-3',
      classifiers=[
          'License :: OSI Approved :: BSD-3 License',
          'Intended Audience :: Developers',
          'Intended Audience :: Scientists',
          'Programming Language :: Python',
          'Topic :: Signal Processing',
      ],
    )
