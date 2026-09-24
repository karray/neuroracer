"""Gymnasium environment for the ROS 2 / Gazebo Jetty racecar.

The caller owns the simulator process. One environment controls one world; this
is not a vectorized environment. All waits have wall-clock deadlines so a paused
or crashed simulator cannot block training forever. Episode time limits come from
the registered TimeLimit wrapper (gym.make(..., max_episode_steps=N)). reset()
places the car at the spawn point with a heading drawn from the seeded np_random,
or at options={'pose': (x, y, yaw)}.
"""
import math
import time
import uuid
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from .observations import image_array, laser_metrics

WHEELBASE = 0.325
STEP_SIZE = 0.001  # racecar_tunnel.sdf max_step_size
PERIOD = 0.1  # Environment step and sensor/odometry update period (model.sdf)
START = (2.0, 3.7, math.pi / 2)  # Spawn point, facing along the tunnel.


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class NeuroRacerEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self, continuous=False, render_mode=None, timeout=30.0):
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Supported render mode: rgb_array')
        self.render_mode = render_mode
        self.continuous = continuous
        self.timeout = timeout
        self.action_space = (gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
                             if continuous else gym.spaces.Discrete(3))
        self.observation_space = gym.spaces.Box(0, 255, (480, 640, 3), np.uint8)
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node('neuroracer_env_' + uuid.uuid4().hex[:8], context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.subscriptions = [
            self.node.create_subscription(Image, '/camera/image_raw', self._camera, 10),
            self.node.create_subscription(LaserScan, '/scan', self._scan, 10),
            self.node.create_subscription(Odometry, '/odom', self._odom, 10),
        ]
        self.control = self.node.create_client(ControlWorld, '/world/racecar_tunnel/control')
        self.set_pose = self.node.create_client(SetEntityPose, '/world/racecar_tunnel/set_pose')
        self.image = self.scan = self.odom = None
        self.image_time = self.scan_time = self.odom_time = -1.0
        self.closed = False

    def _camera(self, msg):
        self.image = image_array(msg)
        self.image_time = stamp_seconds(msg.header.stamp)

    def _scan(self, msg):
        self.scan = np.asarray(msg.ranges, dtype=np.float32)
        self.scan_time = stamp_seconds(msg.header.stamp)

    def _odom(self, msg):
        self.odom = msg.pose.pose
        self.odom_time = stamp_seconds(msg.header.stamp)

    def _stamps(self):
        return self.image_time, self.scan_time, self.odom_time

    def _wait(self, predicate, description):
        deadline = time.monotonic() + self.timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError('Timed out waiting for ' + description + '; start scripts/dev sim or web')
            if not self.context.ok():
                raise RuntimeError('ROS context shut down')
            self.executor.spin_once(timeout_sec=0.05)

    def _call(self, client, request):
        self._wait(client.service_is_ready, client.srv_name)
        future = client.call_async(request)
        self._wait(future.done, client.srv_name + ' response')
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError('Gazebo service failed: ' + client.srv_name)

    def _pause(self, paused):
        request = ControlWorld.Request()
        request.world_control.pause = paused
        self._call(self.control, request)

    def _command(self, steering=0.0, speed=0.0):
        msg = Twist()
        msg.linear.x = float(speed)
        msg.angular.z = speed * math.tan(steering) / WHEELBASE  # Twist carries yaw rate, not steering angle.
        self.publisher.publish(msg)

    def _advance(self):
        # Step exactly one sensor period of physics while paused; unpausing overshoots by
        # the service round-trip, so actions would last a variable, load-dependent time.
        previous = self._stamps()
        request = ControlWorld.Request()
        request.world_control.pause = True
        request.world_control.multi_step = round(PERIOD / STEP_SIZE)
        self._call(self.control, request)
        # Each period holds exactly one 10 Hz camera, lidar and odometry update; nothing
        # else arrives while paused.
        self._wait(lambda: all(now >= before + PERIOD - STEP_SIZE / 2
                               for now, before in zip(self._stamps(), previous)),
                   'camera, lidar, and odometry')

    def _info(self):
        info = {'sim_time': self.image_time}
        if self.odom is not None:
            info['position'] = np.array([self.odom.position.x, self.odom.position.y], dtype=np.float64)
            q = self.odom.orientation
            info['yaw'] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # The paused world publishes sensors only while stepping; unmatched topics would miss them.
        self._wait(lambda: self.publisher.get_subscription_count() > 0
                   and all(s.get_publisher_count() > 0 for s in self.subscriptions), 'ROS/Gazebo bridge')
        # Stop, then teleport. A world reset would recreate plugins without a Reset hook
        # (Ackermann, UserCommands, WebSocket) while their transport callbacks run, which
        # corrupts Gazebo Jetty's heap.
        self._command()
        for _ in range(3):  # Brake from 1 m/s to rest; a teleport keeps velocities.
            self._advance()
        pose = SetEntityPose.Request()
        pose.entity.name = 'racecar'
        pose.entity.type = Entity.MODEL
        if options and 'pose' in options:
            x, y, yaw = map(float, options['pose'])
        else:
            # A random heading varies the first states; the spawn area has over 1 m of
            # clearance all around, so every heading is drivable.
            x, y, yaw = START[0], START[1], START[2] + self.np_random.uniform(-math.pi, math.pi)
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, 0.05
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        self._call(self.set_pose, pose)
        self._advance()
        return self.image, self._info()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Action is outside the action space: {}'.format(action))
        steering = float(action[0]) * 0.6 if self.continuous else (int(action) - 1) * 0.6
        self._command(steering, speed=1.0)
        try:
            self._advance()
        except Exception:
            self._command()
            raise
        reward, terminated, distance = laser_metrics(self.scan)
        return self.image, reward, terminated, False, {**self._info(), 'min_distance': distance}

    def render(self):
        return None if self.image is None else self.image.copy()

    def close(self):
        if self.closed:
            return
        try:
            self._command()
            if self.control.service_is_ready():
                self._pause(True)
        finally:
            self.executor.shutdown()
            self.node.destroy_node()
            self.context.shutdown()
            self.closed = True
