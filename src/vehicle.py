import numpy as np

class Vehicle:
    """
    Vehicle parameters container for constrained steering problem.

    Attributes:
        theta_min (float): Minimum allowed steering angle (radians)
        theta_max (float): Maximum allowed steering angle (radians)
        v_min (float): Minimum allowed speed (can be negative)
        v_max (float): Maximum allowed speed
        l (float): Fixed length of the vector from rear to front wheel
    """
    def __init__(self, theta_min: float, theta_max: float,
                 v_min: float, v_max: float, l: float):
        """
        Initialize the vehicle with steering and velocity constraints.

        Args:
            theta_min (float): Minimum steering angle (radians)
            theta_max (float): Maximum steering angle (radians)
            v_min (float): Minimum speed
            v_max (float): Maximum speed
            l (float): Length of the vector from rear to front wheel
        """
        self.theta_min: float = theta_min
        self.theta_max: float = theta_max
        self.v_min: float = v_min
        self.v_max: float = v_max
        self.l: float = l
