from setuptools import setup

setup(
    name='neuroracer_gym',
    version='0.0.0',
    packages=['neuroracer_gym', 'neuroracer_gym.tasks'],
    package_dir={'': 'src'},
    data_files=[('share/ament_index/resource_index/packages', ['resource/neuroracer_gym']),
                ('share/neuroracer_gym', ['package.xml'])],
    install_requires=['setuptools', 'gymnasium>=1.3,<2', 'numpy'],
    zip_safe=True,
)
