import time

import numpy as np

import rospy
from openai_ros import robot_gazebo_env
from sensor_msgs.msg import LaserScan, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
# from std_msgs.msg import Float64
# from sensor_msgs.msg import Image
# from tf.transformations import quaternion_from_euler

from ackermann_msgs.msg import AckermannDriveStamped

from gym import spaces
from gym.envs.registration import register

# import cv2

timestep_limit_per_episode = 10000 # Can be any Value

default_sleep = 2

register(
        id='NeuroRacer-v0',
        entry_point='neuroracer_gym:neuroracer_env.NeuroRacerEnv',
        timestep_limit=timestep_limit_per_episode,
    )

class NeuroRacerEnv(robot_gazebo_env.RobotGazeboEnv):
    def __init__(self):
        self._init_params()

        self.bridge = CvBridge()

        # Doesnt have any accesibles
        self.controllers_list = []

        # It doesnt use namespace
        self.robot_name_space = ""

        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(NeuroRacerEnv, self).__init__(controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False,
                                            start_init_physics_parameters=False)

        self.gazebo.unpauseSim()
        time.sleep(default_sleep)

        #self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()
        
        self._init_camera()

        self.laser_subscription = rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)
        
        self.drive_control_publisher= rospy.Publisher("/vesc/ackermann_cmd_mux/input/navigation",
                                                       AckermannDriveStamped,
                                                       queue_size=20)

        self._check_publishers_connection()

        self.gazebo.pauseSim()
        
        rospy.logdebug("Finished NeuroRacerEnv INIT...")

    def reset(self):
        params = super(NeuroRacerEnv, self).reset()
        self.gazebo.unpauseSim()
        time.sleep(default_sleep)
        self.gazebo.pauseSim()

        return params

    def _init_params(self):
        self.reward_range = (-np.inf, np.inf)
        self.cumulated_steps = 0.0
        self.last_action = 1
        self.right_left = False

        self.min_distance = .255
        
        # self.steerin_angle_min = -1 # rospy.get_param('neuroracer_env/action_space/steerin_angle_min')
        # self.steerin_angle_max = 1 # rospy.get_param('neuroracer_env/action_space/steerin_angle_max')
        # self.action_space = spaces.Box(low=np.array([self.steerin_angle_min], dtype=np.float32), 
        #                         high=np.array([self.steerin_angle_max], dtype=np.float32))
        self.action_space = spaces.Discrete(3)

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        return True


    # virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        rospy.logdebug("START ALL SENSORS READY")
        # self._check_odom_ready()
        # self._check_imu_ready()
        self._check_laser_scan_ready()
        self._check_camera_ready()
        rospy.logdebug("ALL SENSORS READY")

    # def _check_odom_ready(self):
    #     self.odom = None
    #     rospy.logdebug("Waiting for /odom to be READY...")
    #     while self.odom is None and not rospy.is_shutdown():
    #         try:
    #             self.odom = rospy.wait_for_message("/odom", Odometry, timeout=5.0)
    #             rospy.logdebug("Current /odom READY=>")

    #         except:
    #             rospy.logerr("Current /odom not ready yet, retrying for getting odom")

    #     return self.odom
        
        
    # def _check_imu_ready(self):
    #     self.imu = None
    #     rospy.logdebug("Waiting for /imu to be READY...")
    #     while self.imu is None and not rospy.is_shutdown():
    #         try:
    #             self.imu = rospy.wait_for_message("/imu", Imu, timeout=5.0)
    #             rospy.logdebug("Current /imu READY=>")

    #         except:
    #             rospy.logerr("Current /imu not ready yet, retrying for getting imu")

    #     return self.imu

    def _check_camera_ready(self):
        self.camera_msg = None
        rospy.logdebug("Waiting for /camera/zed/rgb/image_rect_color/compressed to be READY...")
        while self.camera_msg is None and not rospy.is_shutdown():
            try:
                self.camera_msg = rospy.wait_for_message('/camera/zed/rgb/image_rect_color/compressed',
                                          CompressedImage,
                                          timeout=1.0)
            except:
                rospy.logerr("Camera not ready yet, retrying for getting camera_msg")
        
    def _init_camera(self):
        img = self.get_camera_image()

        # self.color_scale = "bgr8" # config["color_scale"]
        self.input_shape = img.shape
        obs_low = 0
        obs_high = 255
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, shape=self.input_shape)

        img_dims = img.shape[0]*img.shape[1]*img.shape[2]
        byte_size = 4
        overhaead = 2 # reserving memory for ros header
        buff_size = img_dims*byte_size*overhaead
        self.camera_msg = rospy.Subscriber("/camera/zed/rgb/image_rect_color/compressed", 
                        CompressedImage, self._camera_callback, queue_size=1, 
                        buff_size=buff_size)
        rospy.logdebug("== Camera READY ==")

    def _check_laser_scan_ready(self):
        self.laser_scan = None
        rospy.logdebug("Waiting for /scan to be READY...")
        while self.laser_scan is None and not rospy.is_shutdown():
            try:
                self.laser_scan = rospy.wait_for_message("/scan", LaserScan, timeout=1.0)
                rospy.logdebug("Current /scan READY=>")

            except:
                rospy.logerr("Current /scan not ready yet, retrying for getting laser_scan")
        return self.laser_scan

    def _get_additional_laser_scan(self):
        laser_scans = []
        # if self.laser_subscription:
        #     self.laser_subscription.unregister()
        self.gazebo.unpauseSim()
        while len(laser_scans) < 2  and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/scan", LaserScan, timeout=1.0)
                laser_scans.append(data.ranges)
            except Exception as e:
                rospy.logerr("getting laser data...")
                print(e)
        self.gazebo.pauseSim()
        # self.laser_subscription = rospy.Subscriber("/scan", LaserScan, self._laser_scan_callback)

        return laser_scans

    def _laser_scan_callback(self, data):
        self.laser_scan = data

    def _camera_callback(self, msg):
        self.camera_msg = msg

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self.drive_control_publisher.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to drive_control_publisher yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("drive_control_publisher Publisher Connected")

        rospy.logdebug("All Publishers READY")
    
    def _set_init_pose(self):
        self.steering(1, speed=0)
        return True
    
    def _init_env_variables(self):
        self.cumulated_reward = 0.0
        self._episode_done = False

    def _compute_reward(self, observations, done):
        reward = -0.001

        if self.right_left:
            reward = -1

        if not done:
            if self.last_action == 1:
                reward = 1
        else:
            reward = -100

        self.cumulated_reward += reward
        self.cumulated_steps += 1
        
        return reward


    def _set_action(self, action):
        steering_angle = 0
        if action == 0:
            steering_angle = -0.7
        if action == 2:
            steering_angle = 0.7

        self.right_left =  action != 1 & self.last_action != 1 & self.last_action != action

        self.last_action = action
        self.steering(steering_angle, speed=10)

    def _get_obs(self):
        return self.get_camera_image()

    def _is_done(self, observations):
        self._episode_done = self._is_collided()
        return self._episode_done
        
    def _create_steering_command(self, steering_angle, speed):
        # steering_angle = np.clip(steering_angle,self.steerin_angle_min, self.steerin_angle_max)
        
        a_d_s = AckermannDriveStamped()
        a_d_s.drive.steering_angle = steering_angle
        a_d_s.drive.steering_angle_velocity = 0.0
        a_d_s.drive.speed = speed  # from 0 to 1
        a_d_s.drive.acceleration = 0.0
        a_d_s.drive.jerk = 0.0

        return a_d_s

    def steering(self, steering_angle, speed=0):
        command = self._create_steering_command(steering_angle, speed)
        self.drive_control_publisher.publish(command)

    # def get_odom(self):
    #     return self.odom
        
    # def get_imu(self):
    #     return self.imu
        
    def get_laser_scan(self):
        return self.laser_scan
    
    def get_camera_image(self):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(self.camera_msg).astype('float32')/255.0
        except Exception as e:
            rospy.logerr("CvBridgeError: Error converting image")
            rospy.logerr(e)
        return cv_image

    def _is_collided(self):
        r = np.array(self.laser_scan.ranges, dtype=np.float32)
        crashed = np.any(r <= self.min_distance)
        if crashed:
            rospy.logdebug('the auto crashed! :(')
            rospy.logdebug('distance: {}'.format(r.min()))
            data = np.array(self._get_additional_laser_scan(), dtype=np.float32)
            data = np.concatenate((np.expand_dims(r, axis=0), data), axis=0)
            data_mean = np.mean(data, axis=0)
            rospy.logdebug('meaned distance: {}'.format(data_mean.min()))

            crashed = np.any(data_mean <= self.min_distance)
            # print(np.where(r <= self.min_distance))
            # print('form', r.shape)


        return crashed
