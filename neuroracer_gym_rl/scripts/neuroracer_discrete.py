from collections import deque
import time

import numpy as np
import gym

import rospy

from neuroracer_gym.tasks import neuroracer_discrete_task
from utils import preprocess, Memory

class NeuroRacer:
    def __init__(self, agent_class, sample_batch_size, n_frames, buffer_max_size, chunk_size, add_flipped, always_explore=False):
        self.sample_batch_size = sample_batch_size
        self.env               = gym.make('NeuroRacer-v0')

        self.highest_reward    = -np.inf
        
        self.n_frames = n_frames

        self.img_y_offset = 200
        self.img_y_scale = 0.2
        self.img_x_scale = 0.2

        state_size = self.env.observation_space.shape
        self.state_size        = (int((state_size[0]-self.img_y_offset)*self.img_y_scale), int(state_size[1]*self.img_x_scale), n_frames)
        rospy.loginfo("State size")
        rospy.loginfo(self.state_size)

        self.action_size       = self.env.action_space.n
        self.agent             = agent_class(self.state_size, self.action_size, buffer_max_size, chunk_size, add_flipped, always_explore=always_explore)

        
    def format_time(self, t):
        m, s = divmod(int(time.time() - t), 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)
    

    
    def run(self):
        total_time = time.time()
        steps = 0
        
        try:
            save_interval = 0
            
            memory = Memory()

            do_training = True
            
            while do_training:
                episode_time = time.time()

                self.env.initial_position = {'p_x': np.random.uniform(1,4), 'p_y': 3.7, 'p_z': 0.05, 'o_x': 0, 'o_y': 0.0, 'o_z': np.random.uniform(0.4,1), 'o_w': 0.855}
                state = self.env.reset()
                state = preprocess(state, self.img_y_offset, self.img_x_scale, self.img_y_scale)
#                 state = np.expand_dims(state, axis=0)

                done = False
                cumulated_reward = 0
                
                stacked_states = deque(maxlen=self.n_frames)
                stacked_next_states = deque(maxlen=self.n_frames)
                for i in range(self.n_frames):
                    stacked_states.append(state)
                    stacked_next_states.append(state)
                    
                episode_steps = 0
                while not done:
                    steps+=1
                    episode_steps+=1
                    
                    action = self.agent.act(np.expand_dims(np.stack(stacked_states, axis=2), axis=0))

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = preprocess(next_state, self.img_y_offset, self.img_x_scale, self.img_y_scale)
                    stacked_next_states.append(next_state)

                    memory.append(action,np.stack(stacked_states, axis=2), np.stack(stacked_next_states, axis=2), reward, done)        
                    stacked_states.append(next_state)
                    
                    cumulated_reward += reward

                    save_interval+=1
                    if save_interval >= self.sample_batch_size:
                        save_interval = 0
                        replay_time = time.time()
                        self.agent.replay(memory)
                        rospy.loginfo("Replay time {}".format(time.time()-replay_time))
                        if steps >= self.sample_batch_size*200:
                            do_training = False
                        memory = Memory()
                    

                if self.highest_reward < cumulated_reward:
                    self.highest_reward = cumulated_reward

                rospy.loginfo("total episode_steps {}, reward {}/{}".format(episode_steps, cumulated_reward, self.highest_reward))
                rospy.loginfo("Episode time {}, total {}".format(self.format_time(episode_time), 
                                                                self.format_time(total_time)))
                rospy.loginfo("exploration_rate {}".format(self.agent.exploration_rate))
                
        finally:
            self.env.close()
            rospy.loginfo("Total time: {}".format(self.format_time(total_time)))
            rospy.loginfo("Total steps: {}".format(steps))
