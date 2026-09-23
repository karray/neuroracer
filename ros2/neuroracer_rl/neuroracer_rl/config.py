from dataclasses import dataclass

ALGORITHMS = ('dqn', 'double_dqn', 'drqn', 'double_drqn', 'ddpg')


@dataclass(frozen=True)
class Config:
    algorithm: str = 'double_dqn'
    frames: int = 16
    height: int = 56
    width: int = 128
    crop_top: int = 200
    batch_size: int = 32
    buffer_size: int = 10000
    warmup: int = 1000
    gamma: float = 0.99
    learning_rate: float = 0.0001
    actor_learning_rate: float = 0.0001
    target_interval: int = 250
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_steps: int = 50000
    noise_std: float = 0.2
    seed: int = 0
    device: str = 'cpu'
    threads: int = 4

    def __post_init__(self):
        if self.algorithm not in ALGORITHMS:
            raise ValueError('Unknown algorithm: ' + self.algorithm)
        # Updates start once replay holds `warmup` transitions; it must be reachable.
        if self.warmup > self.buffer_size:
            raise ValueError('warmup must not exceed buffer_size')

    @property
    def continuous(self):
        return self.algorithm == 'ddpg'

    @property
    def recurrent(self):
        return self.algorithm in ('drqn', 'double_drqn')

    def epsilon(self, step):
        fraction = min(max(step, 0) / self.epsilon_steps, 1.0)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
