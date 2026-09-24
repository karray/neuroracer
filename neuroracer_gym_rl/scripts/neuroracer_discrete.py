from collections import deque
import json
import os
import threading
import time

import numpy as np
import gymnasium as gym

from neuroracer_gym.tasks import neuroracer_discrete_task, neuroracer_continuous_task
from utils import preprocess, loginfo


class Learner(threading.Thread):
    """Replays the agent's buffer continuously while NeuroRacer.run collects, from the
    moment it holds one batch."""
    def __init__(self, agent):
        super(Learner, self).__init__(name='learner', daemon=True)
        self.agent = agent
        self.stopping = threading.Event()
        self.error = None

    def run(self):
        try:
            while not self.stopping.is_set():
                if self.agent.buffer.length() < self.agent.batch_size or self.agent.replay() is False:
                    self.stopping.wait(0.1)
        except BaseException as error:
            self.error = error

    def stop(self):
        self.stopping.set()
        if self.ident is not None:
            self.join()  # Finishes the current replay.


class NeuroRacer:
    def __init__(self, agent_class, sample_batch_size, n_frames, buffer_max_size, chunk_size, add_flipped,
                 env_id='NeuroRacer-v0', working_dir='.', max_episode_steps=1200):
        self.sample_batch_size = sample_batch_size
        # Episodes that reach the limit are truncated: their last state is not terminal.
        self.env               = gym.make(env_id, max_episode_steps=max_episode_steps)

        self.highest_reward    = -np.inf

        self.n_frames = n_frames

        self.img_y_offset = 200
        self.img_size = 224  # resnet18's native input size

        self.state_size        = (self.img_size, self.img_size, n_frames)
        loginfo("State size")
        loginfo(self.state_size)

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_size   = self.env.action_space.n
        else:
            self.action_size   = self.env.action_space.shape[0]
        os.makedirs(working_dir, exist_ok=True)
        self.agent             = agent_class(self.state_size, self.action_size, buffer_max_size, chunk_size, add_flipped,
                                             working_dir=working_dir)
        self.metrics_path      = os.path.join(working_dir, 'metrics.jsonl')


    def format_time(self, t):
        m, s = divmod(int(time.time() - t), 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)



    def run(self, n_steps=None):
        if n_steps is None:
            n_steps = self.sample_batch_size*200
        total_time = time.time()
        steps = 0
        progress = self.agent.progress
        learner = Learner(self.agent)

        try:
            learner.start()
            do_training = True

            while do_training:
                episode_time = time.time()

                yaw = np.random.uniform(-np.pi, np.pi)
                self.env.unwrapped.initial_position = {'p_x': np.random.uniform(1,4), 'p_y': 3.7, 'p_z': 0.05, 'o_x': 0, 'o_y': 0.0, 'o_z': np.sin(yaw/2), 'o_w': np.cos(yaw/2)}
                state, _ = self.env.reset()
                state = preprocess(state, self.img_y_offset, self.img_size)
                self.agent.buffer.start_episode(state)

                done = False
                cumulated_reward = 0

                stacked_states = deque(maxlen=self.n_frames)
                for i in range(self.n_frames):
                    stacked_states.append(state)

                episode_steps = 0
                while not done:
                    if learner.error is not None:
                        raise RuntimeError('Learner failed') from learner.error
                    steps+=1
                    episode_steps+=1

                    # The frames' RGB channels, oldest first, as the replay buffer stacks them.
                    action = self.agent.act(np.expand_dims(np.concatenate(stacked_states, axis=0), axis=0))

                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    next_state = preprocess(next_state, self.img_y_offset, self.img_size)

                    # Only termination (a crash) cuts the return; the stacks are rebuilt from the buffer's frames.
                    self.agent.buffer.append(action, next_state, reward, terminated)
                    stacked_states.append(next_state)

                    cumulated_reward += reward
                    progress['steps'] += 1

                    if steps % self.sample_batch_size == 0:
                        # Once per sample batch, as the original replay did; training runs continuously in Learner.
                        self.agent.save_requested = True
                        if steps >= n_steps:
                            do_training = False


                progress['episodes'] += 1
                if self.highest_reward < cumulated_reward:
                    self.highest_reward = cumulated_reward

                loginfo("total episode_steps {}, reward {}/{}".format(episode_steps, cumulated_reward, self.highest_reward))
                loginfo("Episode time {}, total {}".format(self.format_time(episode_time),
                                                                self.format_time(total_time)))
                loginfo("exploration_rate {}".format(self.agent.exploration_rate))
                with open(self.metrics_path, 'a') as metrics:
                    metrics.write(json.dumps({'step': progress['steps'], 'episode': progress['episodes'],
                                              'steps': episode_steps, 'return': float(cumulated_reward),
                                              'exploration_rate': self.agent.exploration_rate, 'loss': self.agent.loss,
                                              'buffer': self.agent.buffer.length(), 'time': time.time()}) + '\n')

        except KeyboardInterrupt:
            loginfo("Interrupted; waiting for the current replay to finish")
        finally:
            try:
                learner.stop()
                self.agent.save_model()
            finally:
                self.agent.buffer.close()
                self.env.close()
            loginfo("Total time: {}".format(self.format_time(total_time)))
            loginfo("Total steps: {}".format(steps))
