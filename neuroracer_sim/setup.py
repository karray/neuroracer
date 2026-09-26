from pathlib import Path
from setuptools import setup

name = "neuroracer_sim"
data = [("share/ament_index/resource_index/packages", ["resource/" + name]),
        ("share/" + name, ["package.xml"])]
for folder in ("launch", "config", "models", "worlds"):
    for directory in sorted({p.parent for p in Path(folder).rglob("*") if p.is_file()}):
        data.append(("share/" + name + "/" + str(directory),
                     [str(p) for p in sorted(directory.iterdir()) if p.is_file()]))
setup(name=name, version="0.1.0", packages=[], data_files=data,
      install_requires=["setuptools"], zip_safe=True,
      maintainer="aray", maintainer_email="aray@todo.todo", license="TODO",
      description="NeuroRacer Gazebo world, models and launch files")
