import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed

from src.state_space import StateSpace
from src.vehicle import Vehicle

DTYPE = np.float32


class OptimizerTree:
    """
    Tree graph representing discretized vehicle states and feasible steering edges
    for optimal path computation.

    The state of the vehicle is represented as a tuple (x_1, x_2, theta), where
    (x_1, x_2) is the rear wheel position and theta is the orientation of the
    vehicle vector pointing to the front wheel.

    Attributes:
        state_space (StateSpace): The discretized state space of the vehicle.
        vehicle (Vehicle): Vehicle parameters.
        tol (float): Tolerance for checking infinitesimal compatibility.
        nodes (np.ndarray): Array of states in the discretized space (N,3).
        edges_array (np.ndarray): Array of feasible edges (num_edges,7):
            [x_from, y_from, theta_from, x_to, y_to, theta_to, cost]
    """

    def __init__(self, state_space: StateSpace, vehicle: Vehicle, tol: float = 1e-2):
        self.state_space: StateSpace = state_space
        self.vehicle: Vehicle = vehicle
        self.tol: float = tol
        # Convert states list to float32 NumPy array
        self.nodes: np.ndarray = np.array(state_space.states, dtype=DTYPE)
        self.edges_array: np.ndarray = np.empty((0, 7), dtype=DTYPE)

    @staticmethod
    def project(u: np.ndarray, v: np.ndarray, l: float) -> np.ndarray:
        """
        Project vector u onto vector v.
        """
        return np.dot(u, v) / (l**2) * v

    def feasible_steering(
        self,
        state_from: Tuple[float, float, float],
        state_to: Tuple[float, float, float],
        num_mag_samples: int = 50,
        num_angle_samples: int = 25,
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if there exists a feasible steering vector u connecting two states
        under infinitesimal compatibility, steering, and velocity constraints.

        Candidate u vectors are sampled over:
            - magnitudes in [u_min, u_max]
            - angles in [theta_min, theta_max]

        Returns:
            (feasible, cost) where cost is the minimal total wheel path
        """
        x0, y0, theta0 = state_from
        x1, y1, theta1 = state_to

        dx = np.array([x1 - x0, y1 - y0], dtype=DTYPE)
        v0 = np.array([np.cos(theta0), np.sin(theta0)], dtype=DTYPE) * self.vehicle.l

        # Discretized magnitudes and angles
        u_mags = np.linspace(
            self.vehicle.u_min, self.vehicle.u_max, num_mag_samples, dtype=DTYPE
        )
        u_angles = np.linspace(
            self.vehicle.theta_min,
            self.vehicle.theta_max,
            num_angle_samples,
            dtype=DTYPE,
        )

        # Create candidate u vectors (mag x angle)
        u_candidates = np.zeros((num_mag_samples, num_angle_samples, 2), dtype=DTYPE)
        for i, mag in enumerate(u_mags):
            for j, ang in enumerate(u_angles):
                rot = np.array(
                    [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]],
                    dtype=DTYPE,
                )
                u_candidates[i, j, :] = mag * (rot @ v0)

        # Infinitesimal compatibility: dx = projection of u onto v
        dx_proj = (
            np.sum(u_candidates * v0, axis=-1, keepdims=True) / np.dot(v0, v0)
        ) * v0
        compat_mask = np.linalg.norm(dx - dx_proj, axis=-1) <= self.tol

        # Steering angle mask
        cos_angles = np.sum(u_candidates * v0, axis=-1) / (
            np.linalg.norm(u_candidates, axis=-1) * np.linalg.norm(v0) + 1e-12
        )
        angles = np.arccos(np.clip(cos_angles, -1, 1))
        angle_mask = (angles >= self.vehicle.theta_min) & (
            angles <= self.vehicle.theta_max
        )

        # Velocity magnitude mask
        u_norms = np.linalg.norm(u_candidates, axis=-1)
        mag_mask = (u_norms >= self.vehicle.u_min) & (u_norms <= self.vehicle.u_max)

        # Combine masks
        final_mask = compat_mask & angle_mask & mag_mask
        if not np.any(final_mask):
            return False, None

        feasible_u = u_candidates[final_mask]
        dx_norm_sq = np.sum(dx**2)
        feasible_costs = np.sqrt(dx_norm_sq + np.sum(feasible_u**2, axis=-1))

        return True, float(np.min(feasible_costs))

    def build_tree(self, n_jobs: int = -1) -> None:
        """
        Build adjacency edges considering only neighboring states (±1 in each grid dimension).
        Parallelized over idx_from states using joblib. Progress bar shows completed idx_from iterations.
        """
        # Grid dimensions
        N1, N2, Ntheta = (
            len(self.state_space.x1_values),
            len(self.state_space.x2_values),
            len(self.state_space.theta_values),
        )
        total_states = N1 * N2 * Ntheta

        # Helper to convert flat index to grid indices
        def idx_to_grid(idx):
            i1 = idx // (N2 * Ntheta)
            rem = idx % (N2 * Ntheta)
            i2 = rem // Ntheta
            itheta = rem % Ntheta
            return i1, i2, itheta

        # Function to process a single state_from index
        def process_idx_from(idx_from):
            edges_local = []
            i1, i2, itheta = idx_to_grid(idx_from)

            neighbor_i1 = [i for i in [i1 - 1, i1, i1 + 1] if 0 <= i < N1]
            neighbor_i2 = [i for i in [i2 - 1, i2, i2 + 1] if 0 <= i < N2]
            neighbor_theta = [
                i for i in [itheta - 1, itheta, itheta + 1] if 0 <= i < Ntheta
            ]

            for ni1 in neighbor_i1:
                for ni2 in neighbor_i2:
                    for ntheta in neighbor_theta:
                        idx_to = ni1 * N2 * Ntheta + ni2 * Ntheta + ntheta
                        if idx_to == idx_from:
                            continue
                        state_from = self.nodes[idx_from]
                        state_to = self.nodes[idx_to]
                        feasible, cost = self.feasible_steering(
                            tuple(state_from), tuple(state_to)
                        )
                        if feasible:
                            row = np.hstack([state_from, state_to, cost]).astype(DTYPE)
                            edges_local.append(row)
            return edges_local

        # Parallel processing over all idx_from
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_idx_from)(idx_from)
            for idx_from in tqdm(range(total_states), desc="Building tree")
        )

        # Combine all edges
        if results:
            self.edges_array = np.vstack(
                [row for sublist in results for row in sublist]
            )
        else:
            self.edges_array = np.empty((0, 7), dtype=DTYPE)

        print(f"Tree built: {self.edges_array.shape[0]} edges stored as float32")
