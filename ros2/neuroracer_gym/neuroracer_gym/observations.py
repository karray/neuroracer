"""Pure observation helpers shared by the ROS 2 environment and tests."""
import numpy as np


def image_array(message):
    if message.encoding not in ('rgb8', 'bgr8'):
        raise ValueError('Unsupported camera encoding: ' + message.encoding)
    # Image.step includes optional row padding; never reshape the entire buffer blindly.
    rows = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, :message.width * 3].reshape(message.height, message.width, 3)
    if message.encoding == 'bgr8':
        image = image[:, :, ::-1]
    return image.copy()


def laser_metrics(ranges):
    """Reward, collision flag and minimum distance for the 1081-ray lidar."""
    scan = np.minimum(np.asarray(ranges, dtype=np.float32), 10.0)  # +inf means no return.
    right, left, forward = (float(scan[a:b].mean()) for a, b in ((175, 185), (895, 905), (525, 555)))
    # Evaluate all ten-ray neighborhoods; argmin selects the edge of a flat
    # obstacle and can otherwise miss a whole run of equally close returns.
    collision = bool(np.convolve(scan, np.ones(10) / 10, mode='valid').min() <= 0.255)
    reward = -100.0 if collision else (forward - 3.0) - abs(left - right)
    return reward, collision, float(scan.min())
