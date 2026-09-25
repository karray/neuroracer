import math
import time
import uuid

import numpy as np

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import ControlWorld, SetEntityPose

from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from geometry_msgs.msg import Twist

import gymnasium as gym
from gymnasium import spaces

default_timeout = 30.0

WHEELBASE = 0.325
WHEEL_RADIUS = 0.05
STEP_SIZE = 0.001  # racecar_tunnel.sdf max_step_size
PERIOD = 0.1  # model.sdf sensor update period
# Start points (x, y) in racecar_tunnel.sdf; each episode starts at one of them with a random heading.
START_POINTS = ((1.0, 3.7), (2.0, 3.7), (3.0, 3.7), (4.0, 3.7))

class NeuroRacerEnv(gym.Env):
    def __init__(self):

        self.initial_position = None

        self.min_distance = .255

        self.bridge = CvBridge()

        self.timeout = default_timeout
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node('neuroracer_env_' + uuid.uuid4().hex[:8], context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)

        self.control = self.node.create_client(ControlWorld, '/world/racecar_tunnel/control')
        self.set_model_state = self.node.create_client(SetEntityPose, '/world/racecar_tunnel/set_pose')

        self.camera_msg = self.laser_scan = self.odom = None
        self.camera_time = self.laser_time = self.odom_time = -1.0
        self.subscriptions = [
            self.node.create_subscription(Image, '/camera/image_raw', self._camera_callback, 1),
            self.node.create_subscription(LaserScan, '/scan', self._laser_scan_callback, 1),
            self.node.create_subscription(Odometry, '/odom', self._odom_callback, 1),
        ]

        self.drive_control_publisher = self.node.create_publisher(Twist, '/cmd_vel', 20)
        self.closed = False

        self._check_publishers_connection()

        self._check_all_sensors_ready()

        self._init_camera()

        self.node.get_logger().debug("Finished NeuroRacerEnv INIT...")

    def reset_position(self):
        # Teleport: a Gazebo Jetty world reset recreates plugins without a Reset hook.
        self.start = None
        position = self.initial_position or self._random_start()
        q = np.array([position['o_x'], position['o_y'], position['o_z'], position['o_w']], dtype=np.float64)
        q /= np.linalg.norm(q)
        state_msg = SetEntityPose.Request()
        state_msg.entity.name = 'racecar'
        state_msg.entity.type = Entity.MODEL
        state_msg.pose.position.x = float(position['p_x'])
        state_msg.pose.position.y = float(position['p_y'])
        state_msg.pose.position.z = float(position['p_z'])
        state_msg.pose.orientation.x, state_msg.pose.orientation.y, \
            state_msg.pose.orientation.z, state_msg.pose.orientation.w = map(float, q)

        self._call(self.set_model_state, state_msg)

    def _random_start(self):
        self.start = int(self.np_random.integers(len(START_POINTS)))
        x, y = START_POINTS[self.start]
        yaw = self.np_random.uniform(-math.pi, math.pi)
        return {'p_x': x, 'p_y': y, 'p_z': 0.05, 'o_x': 0.0, 'o_y': 0.0, 'o_z': math.sin(yaw / 2), 'o_w': math.cos(yaw / 2)}

    def reset(self, *, seed=None, options=None):
        super(NeuroRacerEnv, self).reset(seed=seed)
        self._check_publishers_connection()
        self._set_init_pose()
        for _ in range(3):  # A teleport keeps velocities, so brake first.
            self._advance()
        self.reset_position()
        self._advance()
        self._init_env_variables()

        return self._get_obs(), {**self._info(), 'start': self.start}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError('Action is outside the action space: {}'.format(action))
        self._set_action(action)
        try:
            self._advance()
        except Exception:
            self.steering(0, speed=0)
            raise
        obs = self._get_obs()
        done = self._is_done(obs)
        reward = self._compute_reward(obs, done)
        return obs, reward, done, False, self._info()

    def close(self):
        if self.closed:
            return
        try:
            self.steering(0, speed=0)
            if self.control.service_is_ready():
                self._pause()
        finally:
            self.executor.shutdown()
            self.node.destroy_node()
            self.context.shutdown()
            self.closed = True

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

    def _pause(self):
        request = ControlWorld.Request()
        request.world_control.pause = True
        self._call(self.control, request)

    def _stamps(self):
        return self.camera_time, self.laser_time, self.odom_time

    def _advance(self):
        # Stepping while paused makes every action last exactly one period.
        previous = self._stamps()
        request = ControlWorld.Request()
        request.world_control.pause = True
        request.world_control.multi_step = round(PERIOD / STEP_SIZE)
        self._call(self.control, request)
        self._wait(lambda: all(now >= before + PERIOD - STEP_SIZE / 2
                               for now, before in zip(self._stamps(), previous)),
                   'camera, lidar, and odometry')

    def _info(self):
        info = {'sim_time': self.camera_time}
        if self.odom is not None:
            info['position'] = np.array([self.odom.position.x, self.odom.position.y], dtype=np.float64)
            q = self.odom.orientation
            info['yaw'] = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return info

    # virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        self.node.get_logger().debug("START ALL SENSORS READY")
        # The paused world publishes sensors only while it advances.
        self._advance()
        self.node.get_logger().debug("ALL SENSORS READY")

    def _init_camera(self):
        img = self.get_camera_image()

        self.input_shape = img.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=self.input_shape, dtype=np.uint8)
        self.node.get_logger().debug("== Camera READY ==")

    def _laser_scan_callback(self, data):
        self.laser_scan = data
        self.laser_time = data.header.stamp.sec + data.header.stamp.nanosec * 1e-9

    def _camera_callback(self, msg):
        self.camera_msg = msg
        self.camera_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _odom_callback(self, msg):
        self.odom = msg.pose.pose
        self.odom_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        self._wait(lambda: self.drive_control_publisher.get_subscription_count() > 0
                   and all(s.get_publisher_count() > 0 for s in self.subscriptions), 'ROS/Gazebo bridge')
        self.node.get_logger().debug("All Publishers READY")

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        return self.get_camera_image()

    def _is_done(self, observations):
        self._episode_done = self._is_collided()
        return self._episode_done

    def _create_steering_command(self, steering_angle, speed):
        # racecar_control's servo_commands.py: wheel rate = speed / 0.1.
        command = Twist()
        command.linear.x = float(speed) / 0.1 * WHEEL_RADIUS
        # angular.z is the yaw rate, not the steering angle.
        command.angular.z = command.linear.x * math.tan(float(steering_angle)) / WHEELBASE

        return command

    def steering(self, steering_angle, speed):
        command = self._create_steering_command(steering_angle, speed)
        self.drive_control_publisher.publish(command)

    def get_laser_scan(self):
        return np.array(self.laser_scan.ranges, dtype=np.float32)

    def get_camera_image(self):
        return self.bridge.imgmsg_to_cv2(self.camera_msg, desired_encoding='bgr8')

    def _is_collided(self):
        r = self.get_laser_scan()
        crashed = np.any(r <= self.min_distance)
        if crashed:
            min_range_idx = r.argmin()
            min_idx = min_range_idx - 5
            if min_idx < 0:
                min_idx = 0
            max_idx = min_idx + 10
            if max_idx >= r.shape[0]:
                max_idx = r.shape[0] - 1
                min_idx = max_idx - 10
            mean_distance = r[min_idx:max_idx].mean()

            crashed = np.any(mean_distance <= self.min_distance)

        return crashed
