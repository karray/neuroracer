from setuptools import find_packages, setup

setup(name='neuroracer_gym', version='0.1.0', packages=find_packages(),
      data_files=[('share/ament_index/resource_index/packages', ['resource/neuroracer_gym']),
                  ('share/neuroracer_gym', ['package.xml'])],
      install_requires=['setuptools', 'gymnasium>=1.3,<2', 'numpy'], zip_safe=True,
      maintainer='aray', maintainer_email='aray@todo.todo', license='TODO',
      description='Gymnasium environment for the ROS 2 NeuroRacer simulator')
