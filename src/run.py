import math

from src.state_space import StateSpace
from src.vehicle import Vehicle
from src.optimizer_tree import OptimizerTree


def main():
    # Define vehicle parameters
    vehicle = Vehicle(
        theta_min=-0.5,  # radians (~-28.6°)
        theta_max=0.5,  # radians (~28.6°)
        u_min=-0.1,
        u_max=0.5,
        l=0.5,
    )

    # Define discretized state space
    state_space = StateSpace(
        x1_min=0.0,
        x1_max=5.0,
        x2_min=0.0,
        x2_max=5.0,
        x1_step=0.25,
        x2_step=0.25,
        theta_step=math.pi / 45.0
    )

    # Initialize optimizer tree
    tree = OptimizerTree(state_space=state_space, vehicle=vehicle, tol=1e-2)

    # Build tree
    print(
        "Building the optimizer tree... this may take some time for large state spaces."
    )
    tree.build_tree()
    print("Tree construction complete.")

    # Print some summary info
    print(f"Number of states: {len(state_space.states)}")
    print(f"Number of nodes in tree: {len(tree.nodes)}")
    total_edges = sum(len(v) for v in tree.edges.values())
    print(f"Total feasible edges: {total_edges}")


if __name__ == "__main__":
    main()
