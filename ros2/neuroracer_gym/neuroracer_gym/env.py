"""Gymnasium environment for the ROS 2 / Gazebo Jetty racecar.

The caller owns the simulator process. One environment controls one world; this
is not a vectorized environment. All waits have wall-clock deadlines so a paused
or crashed simulator cannot block training forever. Episode time limits come from
the registered TimeLimit wrapper (gym.make(..., max_episode_steps=N)).
"""
import math
import time
import uuid
import gymnasium as gym
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose
from .observations import image_array, laser_metrics

WHEELBASE = 0.325


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
            self.node.create_subscription(Image, '/camera/image_raw', self._camera, qos_profile_sensor_data),
            self.node.create_subscription(LaserScan, '/scan', self._scan, qos_profile_sensor_data),
            self.node.create_subscription(Odometry, '/odom', self._odom, 10),
            self.node.create_subscription(Clock, '/clock', self._clock, qos_profile_sensor_data),
        ]
        self.control = self.node.create_client(ControlWorld, '/world/racecar_tunnel/control')
        self.set_pose = self.node.create_client(SetEntityPose, '/world/racecar_tunnel/set_pose')
        self.image = self.scan = self.odom = None
        self.image_time = self.scan_time = self.sim_time = -1.0
        self.closed = False

    def _camera(self, msg):
        self.image = image_array(msg)
        self.image_time = stamp_seconds(msg.header.stamp)

    def _scan(self, msg):
        self.scan = np.asarray(msg.ranges, dtype=np.float32)
        self.scan_time = stamp_seconds(msg.header.stamp)

    def _odom(self, msg):
        self.odom = msg.pose.pose

    def _clock(self, msg):
        self.sim_time = stamp_seconds(msg.clock)

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

    def _advance(self, duration):
        start = self.sim_time
        self._pause(False)
        try:
            self._wait(lambda: self.sim_time >= start + duration
                       and self.image_time >= start + duration * 0.5
                       and self.scan_time >= start + duration * 0.5,
                       'fresh camera, lidar, and simulation clock')
        finally:
            self._pause(True)

    def _info(self):
        info = {'sim_time': self.sim_time}
        if self.odom is not None:
            info['position'] = np.array([self.odom.position.x, self.odom.position.y], dtype=np.float64)
            q = self.odom.orientation
            info['yaw'] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._wait(lambda: self.publisher.get_subscription_count() > 0, 'command bridge')
        self._command()
        request = ControlWorld.Request()
        request.world_control.pause = True
        request.world_control.reset.all = True
        self._call(self.control, request)
        # Drain messages from before the world reset, then await a fresh sensor cycle.
        for _ in range(20):
            self.executor.spin_once(timeout_sec=0.0)
        self.image = self.scan = self.odom = None
        self.image_time = self.scan_time = self.sim_time = -1.0
        pose = SetEntityPose.Request()
        pose.entity.name = 'racecar'
        pose.entity.type = Entity.MODEL
        values = (options or {}).get('pose', (2.0, 3.7, math.pi / 2))
        x, y, yaw = map(float, values)
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, 0.05
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        self._call(self.set_pose, pose)
        self._command()
        self._pause(False)
        try:
            self._wait(lambda: self.image is not None and self.scan is not None
                       and self.odom is not None and self.sim_time >= 0.1, 'sensors after reset')
        finally:
            self._pause(True)
        return self.image, self._info()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Action is outside the action space: {}'.format(action))
        steering = float(action[0]) * 0.6 if self.continuous else (int(action) - 1) * 0.6
        self._command(steering, speed=1.0)
        try:
            self._advance(0.1)
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
