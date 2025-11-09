import numpy as np
from typing import List, Tuple


class StateSpace:
    """
    Discretized 2D state space for a vehicle with orientation.

    The state of the vehicle is represented as a tuple (x, y, theta), where
    (x, y) is the rear wheel position and theta is the orientation of the
    vehicle vector pointing to the front wheel.

    Attributes:
        x_values (np.ndarray): Discretized x-coordinates.
        y_values (np.ndarray): Discretized y-coordinates.
        theta_values (np.ndarray): Discretized orientations (0 to 2*pi).
        states (List[Tuple[float, float, float]]): List of all discretized states.
    """
    def __init__(self,
                 x_min: float, x_max: float,
                 y_min: float, y_max: float,
                 x_step: float, y_step: float, theta_step: float):
        """
        Initialize the discretized state space.

        Args:
            x_min (float): Minimum x-coordinate.
            x_max (float): Maximum x-coordinate.
            y_min (float): Minimum y-coordinate.
            y_max (float): Maximum y-coordinate.
            x_step (float): Step size in x.
            y_step (float): Step size in y.
            theta_step (float): Step size in theta (radians).
        """
        self.x_values: np.ndarray = np.arange(x_min, x_max + x_step, x_step)
        self.y_values: np.ndarray = np.arange(y_min, y_max + y_step, y_step)
        self.theta_values: np.ndarray = np.arange(0, 2 * np.pi, theta_step)

        # List of all discretized states: (x, y, theta)
        self.states: List[Tuple[float, float, float]] = [
            (x, y, theta)
            for x in self.x_values
            for y in self.y_values
            for theta in self.theta_values
        ]
