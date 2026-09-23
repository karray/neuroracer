from setuptools import find_packages, setup

setup(name='neuroracer_rl', version='0.1.0', packages=find_packages(),
      data_files=[('share/ament_index/resource_index/packages', ['resource/neuroracer_rl']),
                  ('share/neuroracer_rl', ['package.xml'])],
      install_requires=['setuptools', 'numpy', 'gymnasium>=1.3,<2', 'torch>=2.14,<3'],
      maintainer='aray', maintainer_email='aray@todo.todo', license='TODO',
      description='PyTorch agents for the ROS 2 NeuroRacer environment',
      entry_points={'console_scripts': ['neuroracer-train = neuroracer_rl.cli:main']})
