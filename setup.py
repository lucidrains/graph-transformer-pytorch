from setuptools import setup, find_packages

setup(
  name = 'graph-transformer-pytorch',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'Graph Transformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/graph-transformer-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'graphs'
  ],
  install_requires=[
    'einops>=0.3',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
