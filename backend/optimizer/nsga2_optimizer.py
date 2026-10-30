import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from optimizer.objective_functions import (
    load_model,
    load_digital_twin,
    generate_receiver_grid,
    evaluate_ap_position
)

from optimizer.placement_constraints import (
    build_valid_ap_polygons,
    get_ap_bounds,
    is_valid_ap_location
)


class APPlacementProblem(ElementwiseProblem):

    def __init__(self, digital_twin, model, receivers):

        self.digital_twin = digital_twin
        self.model = model
        self.receivers = receivers

        min_x, max_x, min_y, max_y = get_ap_bounds(
            digital_twin
        )

        self.ap_polygons = build_valid_ap_polygons(
            digital_twin
        )

        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.array([min_x, min_y]),
            xu=np.array([max_x, max_y])
        )

    def _evaluate(self, x, out, *args, **kwargs):

        ap_x = float(x[0])
        ap_y = float(x[1])

        valid = is_valid_ap_location(
            ap_x,
            ap_y,
            self.ap_polygons
        )

        # Constraint:
        # <= 0 means valid
        constraint = 0.0 if valid else 1.0

        if not valid:
            out["F"] = [1e6, 1e6]
            out["G"] = [constraint]
            return

        result = evaluate_ap_position(
            ap_x,
            ap_y,
            self.receivers,
            self.digital_twin,
            self.model
        )

        coverage = result["coverage"]
        average_rssi = result["average_rssi"]

        # pymoo minimizes objectives.
        # Therefore negate metrics that we want to maximize.
        out["F"] = [
            -coverage,
            -average_rssi
        ]

        out["G"] = [constraint]


def main():

    print("=" * 60)
    print("GeoTwin AI Indoor - NSGA-II AP Placement Optimization")
    print("=" * 60)

    print("\nLoading digital twin...")
    digital_twin = load_digital_twin()

    print("Loading XGBoost RSSI model...")
    model = load_model()

    print("Generating receiver grid...")

    receivers = generate_receiver_grid(
        digital_twin,
        resolution=1.0
    )

    print(f"Receiver points: {len(receivers)}")

    min_x, max_x, min_y, max_y = get_ap_bounds(
        digital_twin
    )

    print(
        f"AP search area: "
        f"X={min_x:.1f} to {max_x:.1f}, "
        f"Y={min_y:.1f} to {max_y:.1f}"
    )

    problem = APPlacementProblem(
        digital_twin,
        model,
        receivers
    )

    algorithm = NSGA2(
        pop_size=40,
        eliminate_duplicates=True
    )

    print("\nRunning NSGA-II...")

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", 30),
        seed=42,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"\nNumber of Pareto-optimal solutions: {len(result.X)}")

    print("\nPareto-optimal AP placements:")

    for i, solution in enumerate(result.X):

        ap_x = float(solution[0])
        ap_y = float(solution[1])

        evaluation = evaluate_ap_position(
            ap_x,
            ap_y,
            receivers,
            digital_twin,
            model
        )

        print(
            f"\nSolution {i + 1}:"
        )

        print(
            f"  AP Location: "
            f"({ap_x:.2f}, {ap_y:.2f})"
        )

        print(
            f"  Coverage: "
            f"{evaluation['coverage']:.2f}%"
        )

        print(
            f"  Average RSSI: "
            f"{evaluation['average_rssi']:.2f} dBm"
        )

        print(
            f"  Minimum RSSI: "
            f"{evaluation['minimum_rssi']:.2f} dBm"
        )

    # Select the solution with the highest average RSSI
    best_index = np.argmax([
        evaluate_ap_position(
            float(solution[0]),
            float(solution[1]),
            receivers,
            digital_twin,
            model
        )["average_rssi"]
        for solution in result.X
    ])

    best_solution = result.X[best_index]

    best_x = float(best_solution[0])
    best_y = float(best_solution[1])

    best_result = evaluate_ap_position(
        best_x,
        best_y,
        receivers,
        digital_twin,
        model
    )

    print("\n" + "=" * 60)
    print("RECOMMENDED AP LOCATION")
    print("=" * 60)

    print(
        f"AP X: {best_x:.2f} m"
    )

    print(
        f"AP Y: {best_y:.2f} m"
    )

    print(
        f"Coverage: "
        f"{best_result['coverage']:.2f}%"
    )

    print(
        f"Average RSSI: "
        f"{best_result['average_rssi']:.2f} dBm"
    )

    print(
        f"Minimum RSSI: "
        f"{best_result['minimum_rssi']:.2f} dBm"
    )


if __name__ == "__main__":
    main()