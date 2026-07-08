from setuptools import find_packages
from distutils.core import setup

setup(name='a2_rl_gym',
      version='1.0.3',
      author='Wu Tianyang',
      license="MIT",
      packages=find_packages(),
      author_email='993660140@qq.com',
      description='RL environments for Unitree A2 Robots',
      install_requires=[
            'isaacgym',
            'matplotlib',
            'numpy==1.20',
            'tensorboard==2.14.0',
            'google-auth==2.45.0',
            'mujoco==3.2.3',
            'pyyaml',
            'onnx==1.17.0',
            'pygame'
      ])
