from collections import deque
import json
import os
import signal
import time
import uuid

import numpy as np
import gymnasium as gym
import rclpy
from rclpy.context import Context
from std_msgs.msg import String

from neuroracer_gym.tasks import neuroracer_discrete_task, neuroracer_continuous_task
from utils import context, preprocess, loader, loginfo


class Telemetry():
    """JSON on a ROS topic, which the bridge forwards to GzWeb's telemetry panel."""
    def __init__(self, topic):
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node('telemetry_' + uuid.uuid4().hex[:8], context=self.context)
        self.publisher = self.node.create_publisher(String, topic, 1)

    def publish(self, **values):
        self.publisher.publish(String(data=json.dumps(values)))

    def close(self):
        self.node.destroy_node()
        self.context.shutdown()


class Learner(context.Process):
    """Trains the agent in its own process. The agent's tensors are shared with the car's process,
    which drives with the EMA network; checkpoints are saved with the car's progress.
    Training ends after `n_updates` updates in total."""
    def __init__(self, agent, n_updates, warmup_steps, metrics_path):
        super(Learner, self).__init__(name='learner')
        self.agent = agent
        self.n_updates = n_updates
        self.warmup_steps = max(warmup_steps, agent.batch_size)
        self.metrics_path = metrics_path
        self.progress = context.Array('q', 2)
        self.updates = context.Value('q', agent.progress['updates'], lock=False)
        self.saving = context.Event()
        self.stopping = context.Event()

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # The car's process stops the learner.
        agent, buffer = self.agent, self.agent.buffer
        self.recent = []  # (loss, Q, dropped transitions) of each update since the last save
        telemetry = Telemetry('/telemetry/learner')
        while buffer.count[0] < self.warmup_steps and not self.stopping.wait(0.1):
            telemetry.publish(buffer=int(buffer.count[0]), warmup_steps=self.warmup_steps)
        if not self.stopping.is_set():
            published = time.monotonic(), agent.progress['updates']
            for batch in loader(buffer, agent.batch_size):
                if self.stopping.is_set() or agent.progress['updates'] >= self.n_updates:
                    break
                agent.replay(batch)
                self.updates.value = agent.progress['updates']
                self.recent.append((agent.loss, agent.q, agent.batch_size - len(batch['rewards'])))
                if time.monotonic() - published[0] >= 1:
                    updates = agent.progress['updates']
                    telemetry.publish(updates=updates, epoch=updates // agent.updates_per_epoch, loss=agent.loss,
                                      q=agent.q, updates_per_s=(updates - published[1]) / (time.monotonic() - published[0]),
                                      buffer=int(buffer.count[0]))
                    published = time.monotonic(), updates
                if self.saving.is_set():
                    self._save()
        self.stopping.wait()  # for the car's final progress
        self._save()
        telemetry.close()
        del self.agent  # Releases the GPU memory shared with the car's process, which owns it.

    def _save(self):
        self.saving.clear()
        self.agent.progress.update(steps=self.progress[0], episodes=self.progress[1])
        self.agent.save_model()
        if self.recent:
            loss, q, dropped = np.array(self.recent).T
            with open(self.metrics_path, 'a') as metrics:
                metrics.write(json.dumps({'step': self.progress[0], 'updates': self.agent.progress['updates'],
                                          'loss': loss.mean(), 'q': q.mean(), 'dropped': int(dropped.sum()),
                                          'time': time.time()}) + '\n')
            self.recent = []

    def save(self, progress):
        self.progress[:] = [progress['steps'], progress['episodes']]
        self.saving.set()

    def stop(self, progress):
        if self.pid is not None:
            self.save(progress)
            self.stopping.set()
            self.join()


class NeuroRacer:
    def __init__(self, agent_class, working_dir, env_id='NeuroRacer-v0', n_epochs=250, max_episode_steps=10000,
                 n_frames=1, sample_batch_size=1000, warmup_steps=1000):
        self.sample_batch_size = sample_batch_size
        self.n_epochs          = n_epochs
        self.warmup_steps      = warmup_steps
        self.env               = gym.make(env_id, max_episode_steps=max_episode_steps)

        self.highest_reward    = -np.inf

        self.n_frames = n_frames

        self.img_y_offset = 200
        self.img_y_scale = 0.2
        self.img_x_scale = 0.2

        state_size = self.env.observation_space.shape
        self.state_size        = (int((state_size[0]-self.img_y_offset)*self.img_y_scale), int(state_size[1]*self.img_x_scale), n_frames)
        loginfo("State size")
        loginfo(self.state_size)

        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_size   = self.env.action_space.n
        else:
            self.action_size   = self.env.action_space.shape[0]
        self.agent             = agent_class(self.state_size, self.action_size, working_dir=working_dir)
        self.working_dir       = working_dir


    def format_time(self, t):
        m, s = divmod(int(time.time() - t), 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)



    def run(self):
        total_time = time.time()
        steps = 0
        progress = self.agent.progress
        # Training lasts a number of epochs, so every run learns from the same amount of sampled
        # data, however many steps the car collects meanwhile.
        n_updates = self.n_epochs * self.agent.updates_per_epoch
        learner = Learner(self.agent, n_updates, self.warmup_steps, os.path.join(self.working_dir, 'updates.jsonl'))
        telemetry = Telemetry('/telemetry/car')

        try:
            learner.start()
            while learner.updates.value < n_updates:
                episode_time = time.time()

                state, info = self.env.reset()
                state = preprocess(state, self.img_y_offset, self.img_x_scale, self.img_y_scale)
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

                    action = self.agent.act(np.expand_dims(np.stack(stacked_states, axis=0), axis=0))

                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated or learner.updates.value >= n_updates
                    next_state = preprocess(next_state, self.img_y_offset, self.img_x_scale, self.img_y_scale)

                    self.agent.buffer.append(action, next_state, reward, terminated)
                    stacked_states.append(next_state)

                    cumulated_reward += reward
                    progress['steps'] += 1
                    telemetry.publish(run=os.path.basename(self.working_dir), mode='train', step=progress['steps'],
                                      episode=progress['episodes'] + 1, episode_step=episode_steps,
                                      action=np.asarray(action).tolist(), reward=float(reward), crashed=bool(terminated),
                                      episode_return=float(cumulated_reward), exploration=self.agent.exploration_rate,
                                      start=info.get('start'))

                    if progress['steps'] % self.sample_batch_size == 0:
                        learner.save(progress)


                progress['episodes'] += 1
                if self.highest_reward < cumulated_reward:
                    self.highest_reward = cumulated_reward

                loginfo("total episode_steps {}, reward {}/{}".format(episode_steps, cumulated_reward, self.highest_reward))
                loginfo("Episode time {}, total {}".format(self.format_time(episode_time),
                                                                self.format_time(total_time)))
                loginfo("exploration_rate {}".format(self.agent.exploration_rate))
                with open(os.path.join(self.working_dir, 'episodes.jsonl'), 'a') as metrics:
                    metrics.write(json.dumps({'step': progress['steps'], 'episode': progress['episodes'],
                                              'steps': episode_steps, 'return': float(cumulated_reward),
                                              'crashed': bool(terminated), 'start': info.get('start'),
                                              'time': time.time()}) + '\n')

        except KeyboardInterrupt:
            loginfo("Interrupted; waiting for the current replay to finish")
        finally:
            try:
                learner.stop(progress)
            finally:
                telemetry.close()
                self.env.close()
            loginfo("Total time: {}".format(self.format_time(total_time)))
            loginfo("Total steps: {}".format(steps))

    def drive(self, n_episodes):
        """Drives with the EMA network and no exploration, without training or saving anything."""
        episodes = []
        telemetry = Telemetry('/telemetry/car')
        try:
            while len(episodes) < n_episodes:
                state, info = self.env.reset()
                state = preprocess(state, self.img_y_offset, self.img_x_scale, self.img_y_scale)
                stacked_states = deque([state] * self.n_frames, maxlen=self.n_frames)
                done, episode_steps, cumulated_reward = False, 0, 0
                while not done:
                    action = self.agent.act(np.expand_dims(np.stack(stacked_states, axis=0), axis=0), explore=False)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    stacked_states.append(preprocess(next_state, self.img_y_offset, self.img_x_scale, self.img_y_scale))
                    episode_steps += 1
                    cumulated_reward += reward
                    telemetry.publish(run=os.path.basename(self.working_dir), mode='drive', episode=len(episodes) + 1,
                                      episode_step=episode_steps,
                                      action=np.asarray(action).tolist(), reward=float(reward), crashed=bool(terminated),
                                      episode_return=float(cumulated_reward), start=info.get('start'))
                episodes.append({'steps': episode_steps, 'return': float(cumulated_reward), 'crashed': bool(terminated),
                                 'start': info.get('start')})
                loginfo("episode {}: {}".format(len(episodes), json.dumps(episodes[-1])))
        except KeyboardInterrupt:
            pass
        finally:
            telemetry.close()
            self.env.close()
        if episodes:
            loginfo("{} episodes: mean steps {:.1f}, mean return {:.1f}, crashed {}".format(
                len(episodes), np.mean([e['steps'] for e in episodes]), np.mean([e['return'] for e in episodes]),
                sum(e['crashed'] for e in episodes)))
        return episodes
