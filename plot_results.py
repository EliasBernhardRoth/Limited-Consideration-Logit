"""Create figures and a formatted Table 1 from completed Monte Carlo runs."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg


BASE_DIRECTORY = Path(__file__).resolve().parent
RESULTS_DIRECTORY = BASE_DIRECTORY / "results"
FIGURE_DIRECTORY = BASE_DIRECTORY / "figures" / f"R{cfg.DENSITY_N_REPETITIONS}"
TABLE_DIRECTORY = BASE_DIRECTORY / "tables" / f"R{cfg.TABLE_EXPERIMENT['n_repetitions']}"

MODEL_NAMES = {"full": "FILM", "sampled": "MLM", "true": "WLM"}
MODEL_TITLES = {
    "full": "FILM: full information",
    "sampled": "MLM: sampled alternatives",
    "true": "WLM: true choice sets",
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#7B4AB5")
LINE_STYLES = ("-", "--", "-.", ":")
PARAMETER_NAMES = ["price"] + [f"product_{j}" for j in range(2, cfg.N_PRODUCTS + 1)]
MODEL_SETTINGS = (
    "N_PRODUCTS", "REFERENCE_PRODUCT_ID", "PRICE_COEFFICIENT", "PRODUCT_EFFECTS",
    "TRUE_COEFFICIENTS", "LIST_PRICES", "PRICE_SCALE", "PRICE_RANDOM_WIDTH",
    "CONSIDERATION_PRICE_COEFFICIENT", "CONSIDERATION_INTERCEPTS",
    "UTILITY_SHOCK_STD", "N_SAMPLED_ALTERNATIVES", "BASE_SEED",
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def check_settings(experiment):
    folder = RESULTS_DIRECTORY / experiment["name"]
    repetitions = experiment["n_repetitions"]
    with (folder / f"settings_R{repetitions}.json").open(encoding="utf-8") as file:
        settings = json.load(file)
    current = {name: getattr(cfg, name) for name in MODEL_SETTINGS}
    current = json.loads(json.dumps(current))  # Convert tuples to JSON lists.
    if (settings.get("run_completed") is not True
            or settings.get("experiment") != experiment
            or settings.get("completed_model_estimations") != 3 * repetitions
            or settings.get("model_parameters") != current
            or settings.get("empty_set_policy") not in ("resample", "raise")):
        raise ValueError(f"Incomplete results or settings mismatch: {folder}")
    return folder


def load_density_estimates(experiment):
    folder = check_settings(experiment)
    repetitions = experiment["n_repetitions"]
    rows = read_csv(folder / f"raw_estimates_R{repetitions}.csv")
    estimates = {model: {} for model in MODEL_NAMES}
    if len(rows) != 3 * repetitions:
        raise ValueError(f"Wrong number of estimates in {folder}")
    for row in rows:
        model = row["model"]
        repetition = int(row["repetition"])
        value = float(row["coef_price"])
        if (model not in estimates or not 1 <= repetition <= repetitions
                or repetition in estimates[model]
                or row["experiment"] != experiment["name"]
                or int(row["seed"]) != cfg.simulation_seed(repetition)
                or int(row["n_decisions"]) != experiment["n_consumers"] * experiment["n_markets"]
                or row["converged"].lower() != "true" or not np.isfinite(value)):
            raise ValueError(f"Invalid or duplicate estimate in {folder}")
        estimates[model][repetition] = value
    if any(len(values) != repetitions for values in estimates.values()):
        raise ValueError(f"Missing model repetitions in {folder}")
    return {
        model: np.array([values[r] for r in range(1, repetitions + 1)])
        for model, values in estimates.items()
    }


def r_bandwidth(values):
    """R's bw.nrd0 rule, including its fallback for coincident quartiles."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Density estimation needs at least two finite values.")
    standard_deviation = np.std(values, ddof=1)
    lower_quartile, upper_quartile = np.percentile(values, [25, 75])
    scale = min(standard_deviation, (upper_quartile - lower_quartile) / 1.34)
    if scale <= 0:
        scale = standard_deviation or abs(values[0]) or 1.0
    return 0.9 * scale * values.size ** (-0.2)


def gaussian_density(grid, values):
    # Direct Gaussian mixture; R uses an FFT approximation to the same KDE.
    bandwidth = r_bandwidth(values)
    distances = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * distances ** 2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))


def save_figure(figure, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = folder / f"{name}.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path.relative_to(BASE_DIRECTORY)}")
    plt.close(figure)


def draw_density_panel(axis, model, curves, grid, maximum_density):
    for index, experiment in enumerate(cfg.DENSITY_EXPERIMENTS):
        density = curves[(model, experiment["n_markets"])]
        color = COLORS[index % len(COLORS)]
        axis.fill_between(grid, density, color=color, alpha=0.10)
        axis.plot(grid, density, color=color, linewidth=1.9,
                  linestyle=LINE_STYLES[index % len(LINE_STYLES)],
                  label=f"T = {experiment['n_markets']}")
    axis.axvline(cfg.PRICE_COEFFICIENT, color="#252525", linestyle="--",
                 linewidth=1.3, label=f"True value: {cfg.PRICE_COEFFICIENT:.3f}")
    axis.set(title=MODEL_TITLES[model], xlabel="Estimated price coefficient",
             xlim=(grid[0], grid[-1]), ylim=(0, maximum_density * 1.08))
    axis.grid(axis="y", color="#DCE2E8", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)


def create_density_figures(data):
    all_values = [values for models in data.values() for values in models.values()]
    left = min([cfg.PRICE_COEFFICIENT] + [v.min() - 4 * r_bandwidth(v) for v in all_values])
    right = max([cfg.PRICE_COEFFICIENT] + [v.max() + 4 * r_bandwidth(v) for v in all_values])
    padding = 0.025 * (right - left)
    grid = np.linspace(left - padding, right + padding, 2048)
    curves = {
        (model, markets): gaussian_density(grid, values)
        for markets, models in data.items() for model, values in models.items()
    }
    maximum_density = max(density.max() for density in curves.values())
    subtitle = (f"{cfg.DENSITY_N_CONSUMERS} consumers per market; "
                f"{cfg.DENSITY_N_REPETITIONS} Monte Carlo repetitions per curve")

    figure, axes = plt.subplots(1, 3, figsize=(14, 5.2), sharex=True, sharey=True)
    for axis, model in zip(axes, MODEL_NAMES):
        draw_density_panel(axis, model, curves, grid, maximum_density)
    axes[0].set_ylabel("Density")
    figure.suptitle("Price-coefficient estimates under limited consideration", fontsize=16, y=0.98)
    figure.text(0.5, 0.90, subtitle, ha="center", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=10)
    figure.subplots_adjust(top=0.78, bottom=0.20, wspace=0.10)
    save_figure(figure, FIGURE_DIRECTORY, "price_coefficient_distributions")

    for model in MODEL_NAMES:
        figure, axis = plt.subplots(figsize=(8, 5.3))
        draw_density_panel(axis, model, curves, grid, maximum_density)
        axis.set_ylabel("Density")
        figure.text(0.5, 0.92, subtitle, ha="center", fontsize=10)
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
        figure.subplots_adjust(top=0.82, bottom=0.20)
        save_figure(figure, FIGURE_DIRECTORY, f"density_{model}")


def paper_stars(ratio):
    for threshold, stars in ((2.576, "***"), (1.96, "**"), (1.645, "*")):
        if abs(ratio) >= threshold:
            return stars
    return ""


def load_table_rows():
    experiment = cfg.TABLE_EXPERIMENT
    folder = check_settings(experiment)
    rows = read_csv(folder / f"table_R{experiment['n_repetitions']}.csv")
    by_parameter = {row["parameter"]: row for row in rows}
    if len(rows) != len(PARAMETER_NAMES) or set(by_parameter) != set(PARAMETER_NAMES):
        raise ValueError("Table 1 has missing, extra or duplicate parameters.")
    ordered_rows = [by_parameter[name] for name in PARAMETER_NAMES]
    for row, truth in zip(ordered_rows, cfg.TRUE_COEFFICIENTS):
        if not np.isclose(float(row["true_value"]), truth, rtol=0, atol=1e-12):
            raise ValueError("Table 1 true values do not match config.py.")
        for model in MODEL_NAMES.values():
            estimate = float(row[f"{model}_estimate"])
            standard_error = float(row[f"{model}_mean_se"])
            if not np.all(np.isfinite([estimate, standard_error])) or standard_error <= 0:
                raise ValueError("Table 1 contains invalid estimates or standard errors.")
            if row[f"{model}_stars"] != paper_stars(estimate / standard_error):
                raise ValueError("Table 1 stars do not match the original R convention.")
    return ordered_rows


def create_table(rows):
    experiment = cfg.TABLE_EXPERIMENT
    repetitions = experiment["n_repetitions"]
    subtitle = (f"{experiment['n_consumers']} consumers per market; "
                f"{experiment['n_markets']} markets; {repetitions} Monte Carlo repetitions")
    headers = ["Parameter", "True value", "FILM", "MLM", "WLM"]
    cells = []
    for row in rows:
        label = "Price" if row["parameter"] == "price" else row["parameter"].replace("_", " ").title()
        cell_row = [label, f"{float(row['true_value']):.4f}"]
        for model in MODEL_NAMES.values():
            estimate = float(row[f"{model}_estimate"])
            standard_error = float(row[f"{model}_mean_se"])
            cell_row.append(f"{estimate:.4f}{row[f'{model}_stars']}\n({standard_error:.4f})")
        cells.append(cell_row)

    notes = [
        "Entries are mean estimates. Parentheses contain average model-based standard errors.",
        "These standard errors are not Monte Carlo standard errors of the mean estimates.",
        "Stars follow the original R rule: |mean estimate / mean SE| >= 1.645 (*), 1.96 (**), 2.576 (***).",
        "This star convention is not a formal test of the Monte Carlo mean. Product 1 is normalized to zero.",
        "FILM: full information; MLM: sampled alternatives; WLM: true choice sets.",
    ]
    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    markdown = ["# Table 1: Monte Carlo estimates", "", subtitle, "",
                "| " + " | ".join(headers) + " |", "| :--- | ---: | ---: | ---: | ---: |"]
    for cell_row in cells:
        formatted = [cell.replace("\n", " ").replace("*", "\\*") for cell in cell_row]
        markdown.append("| " + " | ".join(formatted) + " |")
    markdown += [""] + [note.replace("*", "\\*") for note in notes]
    markdown += ["", f"[Full-precision CSV](../../results/{experiment['name']}/table_R{repetitions}.csv)", ""]
    path = TABLE_DIRECTORY / "table_1.md"
    path.write_text("\n".join(markdown), encoding="utf-8")
    print(f"Saved: {path.relative_to(BASE_DIRECTORY)}")

    figure, axis = plt.subplots(figsize=(10.5, 8))
    axis.axis("off")
    figure.suptitle("Table 1: Monte Carlo estimates", fontsize=17, y=0.96)
    figure.text(0.5, 0.91, subtitle, ha="center", fontsize=10)
    table = axis.table(cellText=cells, colLabels=headers, cellLoc="center",
                       colWidths=[0.24, 0.16, 0.20, 0.20, 0.20], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#243B53")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#EFF3F7" if row % 2 else "#FAFBFC")
            if column == 0:
                cell.set_text_props(ha="left")
    figure.subplots_adjust(left=0.05, right=0.95, top=0.86, bottom=0.23)
    figure.text(0.05, 0.18, "\n".join(notes), va="top", fontsize=8.5, linespacing=1.6)
    save_figure(figure, TABLE_DIRECTORY, "table_1")


def main():
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "pdf.fonttype": 42, "axes.titlepad": 12})
    print("Checking saved results...")
    data = {experiment["n_markets"]: load_density_estimates(experiment)
            for experiment in cfg.DENSITY_EXPERIMENTS}
    rows = load_table_rows()
    print("Creating density figures...")
    create_density_figures(data)
    print("Formatting Table 1...")
    create_table(rows)
    print("\nFinished. Figures and Table 1 are ready.")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        raise SystemExit(
            f"Missing input file: {error.filename}\n"
            "Run monte_carlo.py and monte_carlo_density.py first."
        ) from error
