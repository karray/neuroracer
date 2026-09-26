from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, Shutdown, OpaqueFunction, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.event_handlers import OnShutdown
from launch_ros.actions import Node

SHARE = Path(get_package_share_directory('neuroracer_sim'))


def _server(context):
    world = SHARE / 'worlds/racecar_tunnel.sdf'
    actions = []
    if LaunchConfiguration('web').perform(context).lower() == 'true':
        tree = ET.parse(world)
        plugin = ET.SubElement(tree.getroot().find('world'), 'plugin', {
            'filename': '/opt/neuroracer/lib/libneuroracer-websocket-system.so',
            'name': 'gz::sim::systems::WebsocketServer',
        })
        for name, value in {'port': 9002, 'publication_hz': 20,
                            'max_connections': 4, 'queue_size_per_connection': 100}.items():
            ET.SubElement(plugin, name).text = str(value)
        with tempfile.NamedTemporaryFile(prefix='neuroracer-web-', suffix='.sdf', delete=False) as file:
            tree.write(file, encoding='utf-8', xml_declaration=True)
            world = Path(file.name)
        def cleanup(_context):
            world.unlink(missing_ok=True)
            return []
        actions.append(RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)])))
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(get_package_share_directory('ros_gz_sim') + '/launch/gz_sim.launch.py'),
        launch_arguments={
            # EGL renders the robot sensors without an X server. Without -r the world starts paused.
            'gz_args': ['-s --headless-rendering ', str(world)],
            'on_exit_shutdown': 'true',
        }.items()))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('web', default_value='false'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', str(SHARE / 'models')),
        OpaqueFunction(function=_server),
        Node(package='ros_gz_bridge', executable='parameter_bridge',
             parameters=[{'config_file': str(SHARE / 'config/bridge.yaml')}],
             arguments=[
                 '/world/racecar_tunnel/control@ros_gz_interfaces/srv/ControlWorld',
                 '/world/racecar_tunnel/set_pose@ros_gz_interfaces/srv/SetEntityPose',
             ], output='screen', on_exit=Shutdown(reason='ROS/Gazebo bridge stopped')),
    ])
