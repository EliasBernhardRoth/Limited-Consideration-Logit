import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy

import config as cfg
from simulation import simulate_data
from estimation import estimate_model, MODEL_LABELS


OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "results"

PARAMETER_NAMES = ["price"] + [
    f"product_{product}"
    for product in range(2, cfg.N_PRODUCTS + 1)
]

MODEL_SHORT_NAMES = {
    "full": "FILM",
    "sampled": "MLM",
    "true": "WLM",
}

MODEL_SETTINGS = (
    "N_PRODUCTS",
    "REFERENCE_PRODUCT_ID",
    "PRICE_COEFFICIENT",
    "PRODUCT_EFFECTS",
    "TRUE_COEFFICIENTS",
    "LIST_PRICES",
    "PRICE_SCALE",
    "PRICE_RANDOM_WIDTH",
    "CONSIDERATION_PRICE_COEFFICIENT",
    "CONSIDERATION_INTERCEPTS",
    "UTILITY_SHOCK_STD",
    "N_SAMPLED_ALTERNATIVES",
    "BASE_SEED",
)


def write_csv(path, rows):
    if not rows:
        raise ValueError("There are no results to save.")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_settings(path, settings):
    path.write_text(
        json.dumps(settings, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def paper_significance(ratio):
    """Reproduce the star convention in the original R script."""
    if abs(ratio) >= 2.576:
        return "***"
    if abs(ratio) >= 1.96:
        return "**"
    if abs(ratio) >= 1.645:
        return "*"
    return ""


def summarize_model(model, coefficients, standard_errors):
    coefficients = np.asarray(coefficients, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    true_parameters = np.asarray(cfg.TRUE_COEFFICIENTS, dtype=float)

    if (
        coefficients.ndim != 2
        or coefficients.shape[0] < 2
        or coefficients.shape[1] != len(PARAMETER_NAMES)
        or standard_errors.shape != coefficients.shape
        or true_parameters.shape != (len(PARAMETER_NAMES),)
    ):
        raise ValueError(
            "Invalid shapes or too few repetitions for summarizing."
        )

    if (
        not np.all(np.isfinite(coefficients))
        or not np.all(np.isfinite(standard_errors))
        or np.any(standard_errors <= 0)
    ):
        raise ValueError(
            "Estimates must be finite and standard errors positive."
        )

    n_repetitions = coefficients.shape[0]
    means = coefficients.mean(axis=0)

    # Original Table 1: average the standard errors from individual fits.
    mean_standard_errors = standard_errors.mean(axis=0)

    # Separate Monte Carlo uncertainty across repetitions.
    mc_standard_deviations = coefficients.std(axis=0, ddof=1)
    mc_standard_errors = (
        mc_standard_deviations / np.sqrt(n_repetitions)
    )

    biases = means - true_parameters
    rmse = np.sqrt(
        np.mean((coefficients - true_parameters) ** 2, axis=0)
    )

    rows = []

    for index, parameter in enumerate(PARAMETER_NAMES):
        ratio = float(
            means[index] / mean_standard_errors[index]
        )

        rows.append(
            {
                "model": model,
                "parameter": parameter,
                "n_repetitions": n_repetitions,
                "true_value": float(true_parameters[index]),
                "mean_estimate": float(means[index]),
                "mean_standard_error": float(
                    mean_standard_errors[index]
                ),
                "mc_standard_deviation": float(
                    mc_standard_deviations[index]
                ),
                "mc_standard_error": float(
                    mc_standard_errors[index]
                ),
                "bias": float(biases[index]),
                "rmse": float(rmse[index]),
                "paper_ratio": ratio,
                "paper_stars": paper_significance(ratio),
            }
        )

    return rows


def make_comparison_table(summary_rows):
    """Arrange the three models side by side, as in Table 1."""
    lookup = {
        (row["model"], row["parameter"]): row
        for row in summary_rows
    }

    table_rows = []

    for index, parameter in enumerate(PARAMETER_NAMES):
        row = {
            "parameter": parameter,
            "true_value": float(cfg.TRUE_COEFFICIENTS[index]),
        }

        for model, label in MODEL_SHORT_NAMES.items():
            result = lookup[(model, parameter)]
            row[f"{label}_estimate"] = result["mean_estimate"]
            row[f"{label}_mean_se"] = result["mean_standard_error"]
            row[f"{label}_stars"] = result["paper_stars"]

        table_rows.append(row)

    return table_rows


def run_experiment(experiment, empty_set_policy="resample"):
    """Simulate and estimate all three models for every repetition."""
    for key in ("n_consumers", "n_markets", "n_repetitions"):
        if type(experiment[key]) is not int or experiment[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")

    n_repetitions = experiment["n_repetitions"]

    if n_repetitions < 2:
        raise ValueError("At least two repetitions are required.")

    if cfg.REFERENCE_PRODUCT_ID != 1 or cfg.PRODUCT_EFFECTS[0] != 0:
        raise ValueError(
            "Product 1 must be the reference category with effect zero."
        )

    if empty_set_policy not in ("resample", "raise"):
        raise ValueError(
            "empty_set_policy must be 'resample' or 'raise'."
        )

    directory = OUTPUT_DIRECTORY / experiment["name"]
    directory.mkdir(parents=True, exist_ok=True)

    raw_path = directory / f"raw_estimates_R{n_repetitions}.csv"
    summary_path = directory / f"summary_R{n_repetitions}.csv"
    table_path = directory / f"table_R{n_repetitions}.csv"
    settings_path = directory / f"settings_R{n_repetitions}.json"

    settings = {
        "experiment": dict(experiment),
        "model_parameters": {
            name: getattr(cfg, name) for name in MODEL_SETTINGS
        },
        "seed_rule": (
            "BASE_SEED + repetition, starting at repetition 1"
        ),
        "empty_set_policy": empty_set_policy,
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "paper_star_convention": (
            "Absolute mean estimate divided by mean model standard error; "
            "cutoffs 1.645, 1.96 and 2.576. "
            "This reproduces the R presentation, not a formal test "
            "of the Monte Carlo mean."
        ),
        "run_completed": False,
    }

    save_settings(settings_path, settings)

    coefficient_results = {
        model: [] for model in MODEL_LABELS
    }
    standard_error_results = {
        model: [] for model in MODEL_LABELS
    }
    total_redraws = 0

    raw_columns = [
        "experiment",
        "repetition",
        "seed",
        "model",
        "n_decisions",
        "empty_set_redraws",
        "negative_log_likelihood",
        "optimizer",
        "iterations",
        "max_average_gradient",
        "converged",
        *[f"coef_{name}" for name in PARAMETER_NAMES],
        *[f"se_{name}" for name in PARAMETER_NAMES],
    ]

    print(f"Experiment: {experiment['name']}", flush=True)
    print(
        f"Consumers: {experiment['n_consumers']}; "
        f"markets per consumer: {experiment['n_markets']}; "
        f"repetitions: {n_repetitions}",
        flush=True,
    )
    print(
        f"Total model estimations: {3 * n_repetitions}",
        flush=True,
    )

    with raw_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=raw_columns)
        writer.writeheader()
        file.flush()

        for repetition in range(1, n_repetitions + 1):
            seed = cfg.simulation_seed(repetition)

            # All three models use the same simulated choices and prices.
            data = simulate_data(
                n_consumers=experiment["n_consumers"],
                n_markets=experiment["n_markets"],
                seed=seed,
                empty_set_policy=empty_set_policy,
            )

            total_redraws += data["empty_set_redraws"]
            price_estimates = {}

            for model in MODEL_LABELS:
                try:
                    fit = estimate_model(data, model)
                except (ValueError, RuntimeError) as error:
                    raise RuntimeError(
                        f"Failed: experiment={experiment['name']}, "
                        f"repetition={repetition}, seed={seed}, "
                        f"model={model}. {error}"
                    ) from error

                if (
                    not fit["converged"]
                    or fit["coefficient_names"] != PARAMETER_NAMES
                ):
                    raise RuntimeError(
                        "Invalid or unconverged model result."
                    )

                coefficient_results[model].append(
                    fit["coefficients"].copy()
                )
                standard_error_results[model].append(
                    fit["standard_errors"].copy()
                )
                price_estimates[model] = float(
                    fit["coefficients"][0]
                )

                writer.writerow(
                    {
                        "experiment": experiment["name"],
                        "repetition": repetition,
                        "seed": seed,
                        "model": model,
                        "n_decisions": fit["n_decisions"],
                        "empty_set_redraws": data["empty_set_redraws"],
                        "negative_log_likelihood": fit[
                            "negative_log_likelihood"
                        ],
                        "optimizer": fit["optimizer"],
                        "iterations": fit["iterations"],
                        "max_average_gradient": fit[
                            "max_average_gradient"
                        ],
                        "converged": fit["converged"],
                        **{
                            f"coef_{name}": float(value)
                            for name, value in zip(
                                PARAMETER_NAMES,
                                fit["coefficients"],
                            )
                        },
                        **{
                            f"se_{name}": float(value)
                            for name, value in zip(
                                PARAMETER_NAMES,
                                fit["standard_errors"],
                            )
                        },
                    }
                )
                file.flush()

            print(
                f"Repetition {repetition:03d}/{n_repetitions} | "
                f"Price: FILM={price_estimates['full']:.6f}, "
                f"MLM={price_estimates['sampled']:.6f}, "
                f"WLM={price_estimates['true']:.6f}",
                flush=True,
            )

    summary_rows = []

    for model in MODEL_LABELS:
        summary_rows.extend(
            summarize_model(
                model,
                coefficient_results[model],
                standard_error_results[model],
            )
        )

    write_csv(summary_path, summary_rows)
    write_csv(table_path, make_comparison_table(summary_rows))

    settings["total_empty_set_redraws"] = total_redraws
    settings["completed_model_estimations"] = 3 * n_repetitions
    settings["run_completed"] = True

    save_settings(settings_path, settings)

    print("\nMean estimated price coefficients:")
    for row in summary_rows:
        if row["parameter"] == "price":
            label = MODEL_SHORT_NAMES[row["model"]]
            print(f"{label}: {row['mean_estimate']:.6f}")

    print(f"True price coefficient: {cfg.PRICE_COEFFICIENT}")
    print(f"Total empty-set redraws: {total_redraws}")
    print(f"\nIndividual fits: {raw_path}")
    print(f"Summary statistics: {summary_path}")
    print(f"Comparison table: {table_path}")
    print(f"Run settings: {settings_path}")
    print("\nExperiment completed successfully.")

    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the limited-consideration Monte Carlo experiment."
    )
    parser.add_argument(
        "--experiment",
        choices=["table", "demo"],
        default="table",
    )
    args = parser.parse_args()

    experiment = (
        cfg.TABLE_EXPERIMENT
        if args.experiment == "table"
        else cfg.DEMO_EXPERIMENT
    )

    run_experiment(experiment)