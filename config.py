"""Model parameters and experiment settings for limited consideration."""


# Products and true utility parameters
N_PRODUCTS = 10
REFERENCE_PRODUCT_ID = 1

PRICE_COEFFICIENT = -0.173

# Product effects for products 1 through 10.
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

# Estimation order: price, then product effects 2 through 10.
TRUE_COEFFICIENTS = (PRICE_COEFFICIENT, *PRODUCT_EFFECTS[1:])


# Prices: list_price / 3 + 16 * U, with U uniform on [0, 1).
LIST_PRICES = (
    22.98, 24.98, 22.98, 22.98, 24.98,
    25.00, 24.98, 19.98, 24.98, 24.98,
)

PRICE_SCALE = 1.0 / 3.0
PRICE_RANDOM_WIDTH = 16.0


# Consideration probabilities
CONSIDERATION_PRICE_COEFFICIENT = 0.15
CONSIDERATION_INTERCEPTS = (-2.5,) * 5 + (0.0,) * 5


# Standard deviation of the additional normal utility shock.
# Follow the supplied MATLAB code: 0.25.
# The paper instead describes a standard deviation of 1.
UTILITY_SHOCK_STD = 0.25


# Sampled choice set: chosen product plus four other products.
N_SAMPLED_ALTERNATIVES = 4


# Seed for repetition r: BASE_SEED + r, starting with r = 1.
# Matching seed numbers does not guarantee identical MATLAB draws.
BASE_SEED = 22_102_024


# Small demonstration run
DEMO_EXPERIMENT = {
    "name": "demo",
    "n_consumers": 20,
    "n_markets": 20,
    "n_repetitions": 2,
}


# Table 1: settings reported in Section 3.2 of the paper
TABLE_EXPERIMENT = {
    "name": "table_1",
    "n_consumers": 50,
    "n_markets": 100,
    "n_repetitions": 100,
}


# Figure 1: 50 repetitions for each of four market sizes.
# The supplied MATLAB script currently uses the T=200 setting.
DENSITY_MARKET_SIZES = (10, 50, 100, 200)
DENSITY_N_CONSUMERS = 100
DENSITY_N_REPETITIONS = 50

DENSITY_EXPERIMENTS = [
    {
        "name": f"density_T{n_markets}",
        "n_consumers": DENSITY_N_CONSUMERS,
        "n_markets": n_markets,
        "n_repetitions": DENSITY_N_REPETITIONS,
    }
    for n_markets in DENSITY_MARKET_SIZES
]


def simulation_seed(repetition):
    """Return the seed for a repetition numbered from 1."""
    if type(repetition) is not int or repetition < 1:
        raise ValueError("repetition must be a positive integer.")
    return BASE_SEED + repetition


if __name__ == "__main__":
    print("Configuration loaded successfully.")
    print(f"Products: {N_PRODUCTS}")
    print(f"Reference product: {REFERENCE_PRODUCT_ID}")
    print(f"True price coefficient: {PRICE_COEFFICIENT}")
    print(f"Utility shock SD: {UTILITY_SHOCK_STD}")
    print(f"First simulation seed: {simulation_seed(1)}")
    print()

    experiments = [
        DEMO_EXPERIMENT,
        TABLE_EXPERIMENT,
        *DENSITY_EXPERIMENTS,
    ]

    for experiment in experiments:
        print(
            f"{experiment['name']:<14} "
            f"consumers={experiment['n_consumers']:<3} "
            f"markets={experiment['n_markets']:<3} "
            f"repetitions={experiment['n_repetitions']}"
        )