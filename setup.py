from setuptools import setup, find_packages

setup(
  name = 'vit',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='MIT',
  description = 'Vision Traqnsformer',
  author = 'Akarsh Upadhay',
  author_email = 'akarshupadhyayabc@gmail.com',
  url = 'https://github.com/uakarsh/vision-transformer',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'image classification',
    
  ],
  install_requires=[
    'torch>=1.6',
    'torchvision',
    'transformers',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)
