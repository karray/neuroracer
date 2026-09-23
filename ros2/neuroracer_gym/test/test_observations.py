from types import SimpleNamespace
import numpy as np
from neuroracer_gym.observations import image_array, laser_metrics


def test_padded_bgr_image_is_rgb():
    msg = SimpleNamespace(encoding='bgr8', width=1, height=2, step=4,
                          data=bytes([1, 2, 3, 99, 4, 5, 6, 99]))
    assert image_array(msg).tolist() == [[[3, 2, 1]], [[6, 5, 4]]]


def test_empty_lidar_returns_are_safe_and_finite():
    reward, collision, distance = laser_metrics(np.full(1081, np.inf))
    assert (reward, collision, distance) == (7.0, False, 10.0)


def test_collision_requires_neighborhood_not_one_noisy_ray():
    scan = np.full(1081, 5.0)
    scan[540] = 0.1
    assert laser_metrics(scan)[1] is False
    scan[535:545] = 0.1
    assert laser_metrics(scan)[0:2] == (-100.0, True)
