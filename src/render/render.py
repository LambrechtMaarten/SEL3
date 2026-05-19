"""Utility functions for saving videos and post-processing MuJoCo render output."""

from typing import List

import mediapy as media
import numpy as np
from moojoco.environment.base import MuJoCoEnvironmentConfiguration


def save_video(images: np.ndarray, path: str | None = None) -> str | None:
    """Write a sequence of images to a video file using mediapy.

    Args:
        images: Array of frames with shape ``(T, H, W, C)``.
        path: Output file path.  If None the video is not saved.

    Returns:
        None (mediapy does not return a value).
    """
    media.write_video(path=path, images=images)


def post_render(
    render_output: List[np.ndarray], environment_configuration: MuJoCoEnvironmentConfiguration
) -> np.ndarray | None:
    """Post-process raw MuJoCo render output into a single composited frame.

    If multiple cameras are configured, their frames are stacked horizontally
    for each environment.  If multiple environments are rendered, the
    resulting rows are stacked vertically.  The final image is converted from
    RGB to BGR for OpenCV compatibility.

    Args:
        render_output: List of raw frame arrays returned by the MuJoCo
            renderer, length = ``num_envs * num_cameras``.
        environment_configuration: Environment configuration used to
            determine the number of cameras.

    Returns:
        Composited BGR image array, or None if ``render_output`` is None
        (temporary workaround for MuJoCo issue #1379).
    """
    if render_output is None:
        # Temporary workaround until https://github.com/google-deepmind/mujoco/issues/1379 is fixed
        return None

    num_cameras = len(environment_configuration.camera_ids)
    num_envs = len(render_output) // num_cameras

    if num_cameras > 1:
        # Horizontally stack frames of the same environment
        frames_per_env = np.array_split(render_output, num_envs)
        render_output = [np.concatenate(env_frames, axis=1) for env_frames in frames_per_env]

    # Vertically stack frames of different environments
    render_output = np.concatenate(render_output, axis=0)

    return render_output[:, :, ::-1]  # RGB to BGR
