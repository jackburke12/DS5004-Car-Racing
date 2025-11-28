# utils/frame_stack.py
"""
Frame stacking utility for CarRacing-v3.

Maintains a deque of the last k preprocessed frames and returns
them as a (C, H, W) numpy array.

Used by all RL agents (DQN, Double DQN, Dueling DQN, etc.).
"""

from collections import deque
import numpy as np
from .preprocessing import preprocess_frame


class FrameStack:
    def __init__(self, k, img_h=84, img_w=84):
        """
        Args:
            k: number of frames to stack
            img_h: height for preprocessing
            img_w: width for preprocessing
        """
        self.k = k
        self.img_h = img_h
        self.img_w = img_w
        self.deque = deque(maxlen=k)

    def reset(self, first_frame):
        """
        Called at env.reset() to initialize the stack.

        Args:
            first_frame: raw RGB frame from environment

        Returns:
            np.ndarray shape (k, H, W)
        """
        processed = preprocess_frame(first_frame, self.img_h, self.img_w)
        self.deque.clear()
        for _ in range(self.k):
            self.deque.append(processed)
        return np.stack(self.deque, axis=0).astype(np.float32)

    def append(self, frame):
        """
        Add a new frame to the stack.
        Args:
            frame: raw RGB frame from env.step()

        Returns:
            np.ndarray shape (k, H, W)
        """
        processed = preprocess_frame(frame, self.img_h, self.img_w)
        self.deque.append(processed)
        return np.stack(self.deque, axis=0).astype(np.float32)
