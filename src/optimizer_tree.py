from src.state_space import StateSpace
from src.vehicle import Vehicle
from collections import defaultdict
import numpy as np
from typing import Tuple, List, Dict, Optional


class OptimizerTree:
    """
    Tree graph representing discretized vehicle states and feasible steering edges
    for optimal path computation.

    Attributes:
        state_space (StateSpace): The discretized state space of the vehicle.
        vehicle (Vehicle): The vehicle parameters (steering and velocity constraints, length).
        tol (float): Tolerance for checking infinitesimal compatibility.
        nodes (List[Tuple[float, float, float]]): List of states in the discretized space.
        edges (Dict[Tuple[float, float, float], List[Tuple[Tuple[float, float, float], float]]]):
            Adjacency list of edges: node -> [(neighbor_node, cost), ...]
    """
    def __init__(self, state_space: StateSpace, vehicle: Vehicle, tol: float = 1e-2):
        self.state_space: StateSpace = state_space
        self.vehicle: Vehicle = vehicle
        self.tol: float = tol
        self.nodes: List[Tuple[float, float, float]] = state_space.states
        self.edges: Dict[Tuple[float, float, float], List[Tuple[Tuple[float, float, float], float]]] = defaultdict(list)

    @staticmethod
    def project(u: np.ndarray, v: np.ndarray, l: float) -> np.ndarray:
        """
        Project vector u onto vector v.

        Args:
            u (np.ndarray): Steering vector.
            v (np.ndarray): Vehicle vector.
            l (float): Vehicle vector length.

        Returns:
            np.ndarray: Projection of u onto v.
        """
        return np.dot(u, v) / (l ** 2) * v

    def feasible_steering(self,
                          state_from: Tuple[float, float, float],
                          state_to: Tuple[float, float, float]
                          ) -> Tuple[bool, Optional[float]]:
        """
        Check if there exists a feasible steering vector u connecting two states.

        Args:
            state_from (Tuple[float, float, float]): Starting state (x, y, theta).
            state_to (Tuple[float, float, float]): Target state (x, y, theta).

        Returns:
            Tuple[bool, Optional[float]]: (feasible, cost) where `cost` is the total wheel path
                                          if feasible, else None.
        """
        x0, y0, theta0 = state_from
        x1, y1, theta1 = state_to

        # Rear wheel displacement
        dx = np.array([x1 - x0, y1 - y0])

        # Vehicle vector v (fixed length, from rear to front wheel)
        v = np.array([np.cos(theta0), np.sin(theta0)]) * self.vehicle.l

        # Compute required steering u (initial guess)
        u = dx  # simple initial guess

        # Project u onto v
        dx_proj = self.project(u, v, self.vehicle.l)

        # Check if projection matches dx within tolerance
        if np.linalg.norm(dx - dx_proj) <= self.tol:
            # Check that angle between u and v is within steering limits
            angle = np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1, 1))
            if self.vehicle.theta_min <= angle <= self.vehicle.theta_max:
                # Compute cost: total wheel path (dx + perpendicular component)
                cost = np.linalg.norm(dx) + np.linalg.norm(u - dx_proj)
                return True, cost

        return False, None

    def build_tree(self) -> None:
        """
        Build adjacency edges for all pairs of states in the discretized space.

        Each edge stores the cost of steering from state_from to state_to.
        """
        for i, state_from in enumerate(self.nodes):
            for j, state_to in enumerate(self.nodes):
                if i == j:
                    continue
                feasible, cost = self.feasible_steering(state_from, state_to)
                if feasible:
                    self.edges[state_from].append((state_to, cost))
