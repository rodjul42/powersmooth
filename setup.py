from setuptools import setup

setup(
  name='powersmooth',
  version='0.1.0',
  author='author',
  author_email='aac@example.com',
  packages=['powersmooth'],
  url='http://pypi.python.org/pypi/powersmooth/',
  license='LICENSE.txt',
  description='Smooth noisy time-series faithfully, without distorting lower-order time-derivatives',
  long_description=open('README.md').read(),
  install_requires=[
      "scipy >= 1.7.0"
  ],
)
