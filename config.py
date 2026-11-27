"""Model parameters and experiment settings for limited consideration."""


# Products and utility parameters
N_PRODUCTS = 10
REFERENCE_PRODUCT_ID = 1

PRICE_COEFFICIENT = -0.173

# Product 1 is the reference category.
PRODUCT_EFFECTS = (
    0.0,
    0.368,
    -0.497,
    0.093,
    0.088,
    0.306,
    -0.581,
    -1.075,
    -0.909,
    -0.405,
)

# Estimation order: price, then product effects 2 to 10.
TRUE_COEFFICIENTS = (PRICE_COEFFICIENT,) + PRODUCT_EFFECTS[1:]


# Price generation: list_price / 3 + 16 * U
LIST_PRICES = (
    22.98, 24.98, 22.98, 22.98, 24.98,
    25.00, 24.98, 19.98, 24.98, 24.98,
)

PRICE_SCALE = 1.0 / 3.0
PRICE_RANDOM_WIDTH = 16.0


# Consideration probabilities
CONSIDERATION_PRICE_COEFFICIENT = 0.15
CONSIDERATION_INTERCEPTS = (-2.5,) * 5 + (0.0,) * 5


# Follow the original MATLAB code (0.25 rather than 1).
UTILITY_SHOCK_STD = 0.25


# Chosen product plus four randomly sampled alternatives.
N_SAMPLED_ALTERNATIVES = 4


# Seed for repetition r: BASE_SEED + r.
# Identical seed numbers do not imply identical MATLAB draws.
BASE_SEED = 22_102_024


# Small demonstration experiment
DEMO_EXPERIMENT = {
    "name": "demo",
    "n_consumers": 20,
    "n_markets": 20,
    "n_repetitions": 2,
}


# Table 1 experiment
TABLE_EXPERIMENT = {
    "name": "table_1",
    "n_consumers": 50,
    "n_markets": 100,
    "n_repetitions": 100,
}


# Figure 1 experiments
DENSITY_MARKET_SIZES = (10, 50, 100, 200)
DENSITY_N_CONSUMERS = 100
DENSITY_N_REPETITIONS = 50

DENSITY_EXPERIMENTS = []

for n_markets in DENSITY_MARKET_SIZES:
    experiment = {
        "name": f"density_T{n_markets}",
        "n_consumers": DENSITY_N_CONSUMERS,
        "n_markets": n_markets,
        "n_repetitions": DENSITY_N_REPETITIONS,
    }

    DENSITY_EXPERIMENTS.append(experiment)


def simulation_seed(repetition):
    """Return the seed for a repetition numbered from 1."""

    if repetition < 1:
        raise ValueError("Repetition must be positive.")

    return BASE_SEED + repetition


if __name__ == "__main__":

    print("Configuration loaded successfully.")
    print(f"Products: {N_PRODUCTS}")
    print(f"Reference product: {REFERENCE_PRODUCT_ID}")
    print(f"True price coefficient: {PRICE_COEFFICIENT}")
    print(f"Utility shock SD: {UTILITY_SHOCK_STD}")
    print(f"First simulation seed: {simulation_seed(1)}")
    print()

    experiments = [DEMO_EXPERIMENT, TABLE_EXPERIMENT]
    experiments += DENSITY_EXPERIMENTS

    for experiment in experiments:
        print(
            f"{experiment['name']:<14} "
            f"consumers={experiment['n_consumers']:<3} "
            f"markets={experiment['n_markets']:<3} "
            f"repetitions={experiment['n_repetitions']}"
        )