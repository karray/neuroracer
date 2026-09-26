# Model provenance

The racecar meshes and MIT_Tunnel geometry come from
https://github.com/mit-racecar/racecar-simulator at commit
`effd29f9edc20a03c8a1f666fb6fb6c0cbbee8ff`. The upstream package declares its
license as `TODO`; this project does not assign a new license to those assets.

The racecar SDF was exported from the upstream xacro. Masses, joint locations,
wheel geometry, camera geometry and lidar field of view are retained. Sensors and
drive use Gazebo Sim's native sensors and Ackermann steering system (steering
limited to 1 rad, speed to 5 m/s).

Gazebo Sim cannot read Gazebo Classic material scripts, so they are reproduced
inline: `Gazebo/Grey` as its colors, and `Gazebo/CeilingTiled` as a matte,
non-metallic PBR albedo map of `ceiling_tiled.jpg`, copied from
https://github.com/gazebosim/gazebo-classic (`media/materials/textures`, Apache-2.0).
