import numpy as np
from typing import List, Tuple


class StateSpace:
    """
    Discretized 2D state space for a vehicle with orientation.

    The state of the vehicle is represented as a tuple (x_1, x_2, theta), where
    (x_1, x_2) is the rear wheel position and theta is the orientation of the
    vehicle vector pointing to the front wheel.

    Attributes:
        x1_values (np.ndarray): Discretized x_1-coordinates.
        x2_values (np.ndarray): Discretized x_2-coordinates.
        theta_values (np.ndarray): Discretized orientations (0 to 2*pi).
        states (List[Tuple[float, float, float]]): List of all discretized states.
    """

    def __init__(
        self,
        x1_min: float,
        x1_max: float,
        x2_min: float,
        x2_max: float,
        x1_step: float = 0.1,
        x2_step: float = 0.1,
        theta_step: float = np.pi / 16,
    ):
        """
        Initialize the discretized state space.

        Args:
            x1_min (float): Minimum x_1-coordinate.
            x1_max (float): Maximum x_1-coordinate.
            x2_min (float): Minimum x_2-coordinate.
            x2_max (float): Maximum x_2-coordinate.
            x1_step (float): Step size in x_1 (default 0.1).
            x2_step (float): Step size in x_2 (default 0.1).
            theta_step (float): Step size in theta (radians, default pi/16).
        """
        self.x1_values: np.ndarray = np.arange(x1_min, x1_max + x1_step, x1_step, dtype=np.float32)
        self.x2_values: np.ndarray = np.arange(x2_min, x2_max + x2_step, x2_step, dtype=np.float32)
        self.theta_values: np.ndarray = np.arange(0, 2 * np.pi, theta_step, dtype=np.float32)

        # Create a meshgrid and flatten into an (N,3) array
        X1, X2, THETA = np.meshgrid(
            self.x1_values, self.x2_values, self.theta_values, indexing='ij'
        )
        self.states = np.stack([X1.ravel(), X2.ravel(), THETA.ravel()], axis=-1)  # shape (N,3)
