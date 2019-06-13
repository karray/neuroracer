#!/usr/bin/env python

import rospy

from neuroracer_discrete import NeuroRacer

if __name__ == '__main__':

    rospy.init_node('neuroracer_qlearn', anonymous=True, log_level=rospy.INFO)

    always_explore = rospy.get_param('/neuroracer_gym/always_explore')
    agent_name = rospy.get_param('/neuroracer_gym/agent_name')
    
    module = __import__(agent_name)
    agent_class = getattr(module, 'Agent')

    game = NeuroRacer(agent_class, \
                      sample_batch_size=1000, \
                      n_frames=16, \
                      buffer_max_size=100000, \
                      chunk_size=20000, \
                      add_flipped=False)
    rospy.logwarn("Gym environment done. always_explore = " + str(always_explore))
    rospy.logwarn("Agent is " + agent_name)

    # Set the logging system
    # rospack = rospkg.RosPack()
    # pkg_path = rospack.get_path('demo')
    # outdir = pkg_path + '/training_results'
    # rospy.loginfo("Monitor Wrapper started")

    game.run()