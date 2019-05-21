from collections import deque
import time

import numpy as np
import gym

import rospy

from neuroracer_gym.tasks import neuroracer_discrete_task
from utils import preprocess, Memory

class NeuroRacer:
    def __init__(self, agent_class, sample_batch_size=5000, n_frames=8, episodes=10000, always_explore=False):
        self.sample_batch_size = sample_batch_size
        self.episodes          = episodes
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
        self.agent             = agent_class(self.state_size, self.action_size, always_explore=always_explore)

        
    def format_time(self, t):
        m, s = divmod(int(time.time() - t), 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)
    

    
    def run(self):
        try:
            total_time = time.time()
            save_interval = 0
            
            memory = Memory()

            for index_episode in range(self.episodes):
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
                
                steps = 0
                while not done:
                    steps+=1
                    # if index_episode % 50 == 0:
                    #     self.env.render()
                    
                    action = self.agent.act(np.expand_dims(np.stack(stacked_states, axis=2), axis=0))
#                     action = self.agent.act(state)


                    next_state, reward, done, _ = self.env.step(action)
                    next_state = preprocess(next_state, self.img_y_offset, self.img_x_scale, self.img_y_scale)
#                     next_state = np.expand_dims(next_state, axis=0)
                    stacked_next_states.append(next_state)

                    memory.append(action,np.stack(stacked_states, axis=2), np.stack(stacked_next_states, axis=2), reward, done)        
#                     self.agent.remember(history, action, reward, next_history, done)
                    stacked_states.append(next_state)
#                     self.agent.remember(state, action, reward, next_state, done)
#                     state = next_state
                    
                    cumulated_reward += reward

                    if save_interval >= self.sample_batch_size:
                        save_interval = 0
                        replay_time = time.time()
                        self.agent.replay(memory)
                        rospy.loginfo("Replay time {}".format(time.time()-replay_time))
                        memory = Memory()
                        # rospy.loginfo("Episode {} of {}. In-episode training".format(index_episode, self.episodes))
                        # rospy.loginfo("Step {}, reward {}/{}".format(steps, cumulated_reward, self.highest_reward))
                        # rospy.loginfo("Episode time {}, total {}".format(self.format_time(episode_time), 
                        #                                                 self.format_time(total_time)))
                    save_interval+=1
                    

                if self.highest_reward < cumulated_reward:
                    self.highest_reward = cumulated_reward

                rospy.loginfo("Episode {} of {}".format(index_episode, self.episodes))
                rospy.loginfo("total steps {}, reward {}/{}".format(steps, cumulated_reward, self.highest_reward))
                rospy.loginfo("Episode time {}, total {}".format(self.format_time(episode_time), 
                                                                self.format_time(total_time)))
                rospy.loginfo("exploration_rate {}".format(self.agent.exploration_rate))
                
                # self.agent.replay(self.sample_batch_size)
                # if index_episode % 50 == 0:
                #     self.env.close()
        finally:
            self.env.close()
            # self.agent.save_model()
            # print("Best score: " + str(self.best_score))