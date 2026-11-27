import argparse

import config as cfg
from estimation import MODEL_LABELS
from monte_carlo import run_experiment


def run_density_experiments(n_markets=None):
    """Run the market-size experiments used for Figure 1."""

    experiments = []

    # Select the requested market sizes.
    for experiment in cfg.DENSITY_EXPERIMENTS:
        if n_markets is None or experiment["n_markets"] == n_markets:
            experiments.append(experiment)

    if not experiments:
        raise ValueError("No matching density experiment.")

    # Calculate the total number of simulations and estimations.
    total_repetitions = 0

    for experiment in experiments:
        total_repetitions += experiment["n_repetitions"]

    total_fits = len(MODEL_LABELS) * total_repetitions

    print(f"Market-size settings: {len(experiments)}", flush=True)
    print(f"Simulated datasets: {total_repetitions}", flush=True)
    print(f"Total model estimations: {total_fits}", flush=True)

    # Run each selected experiment.
    for index, experiment in enumerate(experiments, start=1):

        print(
            f"\nSetting {index}/{len(experiments)}: "
            f"{experiment['n_markets']} markets per consumer",
            flush=True,
        )

        run_experiment(experiment)

    print("\nAll requested density experiments completed successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the four market-size experiments for Figure 1."
    )

    parser.add_argument(
        "--markets",
        type=int,
        choices=cfg.DENSITY_MARKET_SIZES,
        default=None,
        help="Run one market size. By default, run all four.",
    )

    args = parser.parse_args()

    run_density_experiments(n_markets=args.markets)