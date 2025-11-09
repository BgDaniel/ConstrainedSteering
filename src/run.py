from src.state_space import StateSpace
from src.vehicle import Vehicle
from src.optimizer_tree import OptimizerTree  # renamed from VehicleTree

def main():
    # Define vehicle parameters
    vehicle = Vehicle(theta_min=-0.5, theta_max=0.5, v_min=0.0, v_max=1.0, l=2.0)

    # Define state space
    state_space = StateSpace(
        x_min=0.0, x_max=5.0,
        y_min=0.0, y_max=5.0,
        x_step=1.0, y_step=1.0, theta_step=0.25
    )

    # Build optimizer tree
    tree = OptimizerTree(state_space=state_space, vehicle=vehicle, tol=1e-2)
    tree.build_tree()

    # Print some info
    print(f"Number of states: {len(state_space.states)}")
    print(f"Number of nodes in tree: {len(tree.nodes)}")
    total_edges = sum(len(v) for v in tree.edges.values())
    print(f"Total feasible edges: {total_edges}")

if __name__ == "__main__":
    main()
