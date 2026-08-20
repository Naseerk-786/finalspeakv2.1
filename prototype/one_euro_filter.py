# One-Euro Adaptive Filter for 126-dim Real-Time Hand Landmark Smoothing
# Eliminates webcam sensor jitter at low speeds while avoiding lag during fast transitions.
# Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in HCI" (CHI 2012)

import math
import numpy as np


class LowPassFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.last_val = None

    def filter(self, val):
        if self.last_val is None:
            self.last_val = val
            return val
        res = self.alpha * val + (1.0 - self.alpha) * self.last_val
        self.last_val = res
        return res

    def reset(self):
        self.last_val = None


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, timestamp=None):
        if timestamp is None:
            import time
            timestamp = time.time()

        if self.last_time is None:
            self.last_time = timestamp
            self.x_filter.filter(x)
            return x

        dt = max(1e-4, timestamp - self.last_time)
        self.last_time = timestamp

        # Estimate derivative
        dx = (x - self.x_filter.last_val) / dt
        edx = self.dx_filter.filter(dx)
        self.dx_filter.alpha = self._alpha(self.d_cutoff, dt)

        # Dynamic cutoff frequency based on velocity
        speed = np.abs(edx) if isinstance(edx, np.ndarray) else abs(edx)
        cutoff = self.min_cutoff + self.beta * speed
        
        if isinstance(cutoff, np.ndarray):
            alpha = np.array([self._alpha(c, dt) for c in cutoff], dtype=np.float32)
        else:
            alpha = self._alpha(cutoff, dt)

        self.x_filter.alpha = alpha
        return self.x_filter.filter(x)

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class LandmarkStreamSmoother:
    """Smoothes 126-dimensional MediaPipe hand landmark arrays in real time."""
    def __init__(self, dim=126, min_cutoff=0.8, beta=0.01):
        self.dim = dim
        self.filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

    def smooth(self, feat_vec, timestamp=None):
        if feat_vec is None:
            self.filter.reset()
            return None
        return self.filter.filter(feat_vec, timestamp)

    def reset(self):
        self.filter.reset()
