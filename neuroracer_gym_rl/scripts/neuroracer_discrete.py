from collections import deque
import json
import os
import signal
import time

import numpy as np
import gymnasium as gym

from neuroracer_gym.tasks import neuroracer_discrete_task, neuroracer_continuous_task
from utils import context, preprocess, loader, loginfo


class Learner(context.Process):
    """Trains the agent in its own process. The agent's tensors are shared with the car's process,
    which drives with the EMA network; checkpoints are saved with the car's progress."""
    def __init__(self, agent):
        super(Learner, self).__init__(name='learner')
        self.agent = agent
        self.progress = context.Array('q', 2)
        self.saving = context.Event()
        self.stopping = context.Event()
        self.loss = context.Value('d', float('nan'))

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # The car's process stops the learner.
        while self.agent.buffer.length() < self.agent.batch_size and not self.stopping.wait(0.1):
            pass
        if not self.stopping.is_set():
            for batch in loader(self.agent.buffer, self.agent.batch_size):
                if self.stopping.is_set():
                    break
                if self.agent.replay(batch) is False:
                    self.stopping.wait(0.1)
                    continue
                self.loss.value = self.agent.loss
                if self.saving.is_set():
                    self._save()
        self._save()
        del self.agent  # Releases the GPU memory shared with the car's process, which owns it.

    def _save(self):
        self.saving.clear()
        self.agent.progress = {'steps': self.progress[0], 'episodes': self.progress[1]}
        self.agent.save_model()

    def save(self, progress):
        self.progress[:] = [progress['steps'], progress['episodes']]
        self.saving.set()

    def stop(self, progress):
        if self.pid is not None:
            self.save(progress)
            self.stopping.set()
            self.join()


class NeuroRacer:
    def __init__(self, agent_class, sample_batch_size, n_frames, buffer_max_size, add_flipped,
                 env_id='NeuroRacer-v0', working_dir='.', max_episode_steps=1200):
        self.sample_batch_size = sample_batch_size
        self.env               = gym.make(env_id, max_episode_steps=max_episode_steps)

        self.highest_reward    = -np.inf

        self.n_frames = n_frames

        self.img_y_offset = 200
        self.img_size = 224

        self.state_size        = (self.img_size, self.img_size, n_frames)
        loginfo("State size")
        loginfo(self.state_size)

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_size   = self.env.action_space.n
        else:
            self.action_size   = self.env.action_space.shape[0]
        os.makedirs(working_dir, exist_ok=True)
        self.agent             = agent_class(self.state_size, self.action_size, buffer_max_size, add_flipped,
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
                    if learner.exitcode is not None:
                        raise RuntimeError('Learner failed')
                    steps+=1
                    episode_steps+=1

                    action = self.agent.act(np.expand_dims(np.concatenate(stacked_states, axis=0), axis=0))

                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    next_state = preprocess(next_state, self.img_y_offset, self.img_size)

                    self.agent.buffer.append(action, next_state, reward, terminated)
                    stacked_states.append(next_state)

                    cumulated_reward += reward
                    progress['steps'] += 1

                    if steps % self.sample_batch_size == 0:
                        learner.save(progress)
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
                                              'exploration_rate': self.agent.exploration_rate, 'loss': learner.loss.value,
                                              'buffer': self.agent.buffer.length(), 'time': time.time()}) + '\n')

        except KeyboardInterrupt:
            loginfo("Interrupted; waiting for the current replay to finish")
        finally:
            try:
                learner.stop(progress)
            finally:
                self.agent.buffer.close()
                self.env.close()
            loginfo("Total time: {}".format(self.format_time(total_time)))
            loginfo("Total steps: {}".format(steps))
