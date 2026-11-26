import numpy as np
from scipy.special import expit, softmax

import config as cfg


MAX_CONSIDERATION_DRAWS = 10000


def simulate_data(
    n_consumers,
    n_markets,
    seed,
    empty_set_policy="resample",
):
    """
    Simulate one dataset for the three choice-set models.

    Each row represents one consumer-market decision.
    Product IDs run from 1 to N_PRODUCTS.

    empty_set_policy:
        "resample": redraw consideration at the same prices until nonempty.
        "raise": stop immediately if a consideration set is empty.

    Redrawing empty sets is an explicit extension of the original code.
    """
    for name, value in (
        ("n_consumers", n_consumers),
        ("n_markets", n_markets),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")

    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or not 0 <= seed < 2**32
    ):
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")

    if empty_set_policy not in ("resample", "raise"):
        raise ValueError("empty_set_policy must be 'resample' or 'raise'.")

    n_products = cfg.N_PRODUCTS
    n_sampled = cfg.N_SAMPLED_ALTERNATIVES

    list_prices = np.asarray(cfg.LIST_PRICES, dtype=float)
    product_effects = np.asarray(cfg.PRODUCT_EFFECTS, dtype=float)
    consideration_intercepts = np.asarray(
        cfg.CONSIDERATION_INTERCEPTS,
        dtype=float,
    )

    for values in (list_prices, product_effects, consideration_intercepts):
        if values.shape != (n_products,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Product parameters must contain one finite value per product."
            )

    if (
        isinstance(n_sampled, (bool, np.bool_))
        or not isinstance(n_sampled, (int, np.integer))
        or not 0 <= n_sampled < n_products
    ):
        raise ValueError("Invalid number of sampled alternatives.")

    scalar_parameters = (
        cfg.PRICE_COEFFICIENT,
        cfg.PRICE_SCALE,
        cfg.PRICE_RANDOM_WIDTH,
        cfg.CONSIDERATION_PRICE_COEFFICIENT,
        cfg.UTILITY_SHOCK_STD,
    )

    if not np.all(np.isfinite(scalar_parameters)):
        raise ValueError("Model parameters must be finite.")

    if cfg.PRICE_RANDOM_WIDTH < 0 or cfg.UTILITY_SHOCK_STD < 0:
        raise ValueError(
            "Price variation and shock standard deviation must be nonnegative."
        )

    n_decisions = int(n_consumers) * int(n_markets)
    rng = np.random.RandomState(int(seed))

    prices = np.empty((n_decisions, n_products))
    probabilities = np.zeros((n_decisions, n_products))
    true_set = np.zeros((n_decisions, n_products), dtype=bool)
    sampled_set = np.zeros((n_decisions, n_products), dtype=bool)
    chosen_product = np.empty(n_decisions, dtype=np.int64)

    product_indices = np.arange(n_products)
    empty_set_redraws = 0

    # All markets for consumer 1, then all markets for consumer 2, etc.
    for decision in range(n_decisions):
        current_prices = (
            list_prices * cfg.PRICE_SCALE
            + cfg.PRICE_RANDOM_WIDTH * rng.random_sample(n_products)
        )

        attention = (
            cfg.CONSIDERATION_PRICE_COEFFICIENT * current_prices
            + consideration_intercepts
        )
        consideration_probabilities = expit(attention)

        considered = (
            rng.random_sample(n_products) < consideration_probabilities
        )
        attempts = 1

        while not np.any(considered):
            if empty_set_policy == "raise":
                raise RuntimeError(
                    f"Empty consideration set at decision {decision + 1}, "
                    f"seed={seed}."
                )

            if attempts >= MAX_CONSIDERATION_DRAWS:
                raise RuntimeError(
                    "Could not draw a nonempty consideration set. "
                    "Check the consideration parameters."
                )

            considered = (
                rng.random_sample(n_products) < consideration_probabilities
            )
            empty_set_redraws += 1
            attempts += 1

        utilities = (
            cfg.PRICE_COEFFICIENT * current_prices
            + product_effects
            + cfg.UTILITY_SHOCK_STD * rng.standard_normal(n_products)
        )

        # Only considered products receive positive choice probabilities.
        considered_indices = np.flatnonzero(considered)
        considered_probabilities = softmax(utilities[considered_indices])

        cumulative_probabilities = np.cumsum(considered_probabilities)
        cumulative_probabilities[-1] = 1.0

        position = np.searchsorted(
            cumulative_probabilities,
            rng.random_sample(),
            side="right",
        )
        chosen_index = considered_indices[position]

        # Sample uniformly from ALL other products, including products
        # outside the consumer's true consideration set.
        other_products = product_indices[product_indices != chosen_index]
        sampled_others = rng.choice(
            other_products,
            size=n_sampled,
            replace=False,
        )

        prices[decision] = current_prices
        true_set[decision] = considered
        probabilities[decision, considered_indices] = considered_probabilities
        chosen_product[decision] = chosen_index + 1

        sampled_set[decision, chosen_index] = True
        sampled_set[decision, sampled_others] = True

    return {
        "decision_id": np.arange(1, n_decisions + 1),
        "consumer_id": np.repeat(
            np.arange(1, n_consumers + 1), n_markets
        ),
        "market_id": np.tile(
            np.arange(1, n_markets + 1), n_consumers
        ),
        "prices": prices,
        "probabilities": probabilities,
        "chosen_product": chosen_product,
        "full_set": np.ones_like(true_set),
        "sampled_set": sampled_set,
        "true_set": true_set,
        "seed": int(seed),
        "empty_set_policy": empty_set_policy,
        "empty_set_redraws": empty_set_redraws,
    }


if __name__ == "__main__":
    experiment = cfg.DEMO_EXPERIMENT

    data = simulate_data(
        n_consumers=experiment["n_consumers"],
        n_markets=experiment["n_markets"],
        seed=cfg.simulation_seed(1),
    )

    n_decisions = experiment["n_consumers"] * experiment["n_markets"]
    rows = np.arange(n_decisions)
    chosen_indices = data["chosen_product"] - 1

    assert data["prices"].shape == (n_decisions, cfg.N_PRODUCTS)
    assert np.all(np.isfinite(data["prices"]))
    assert np.all(data["true_set"].sum(axis=1) >= 1)
    assert np.all(data["true_set"][rows, chosen_indices])
    assert np.all(data["sampled_set"][rows, chosen_indices])

    assert np.all(
        data["sampled_set"].sum(axis=1)
        == cfg.N_SAMPLED_ALTERNATIVES + 1
    )

    np.testing.assert_allclose(
        data["probabilities"].sum(axis=1),
        1.0,
        rtol=0,
        atol=1e-12,
    )
    assert np.all(data["probabilities"][~data["true_set"]] == 0)

    print("Simulation completed successfully.")
    print(f"Decisions: {n_decisions}")
    print(f"Price matrix shape: {data['prices'].shape}")
    print(
        "Average true consideration-set size: "
        f"{data['true_set'].sum(axis=1).mean():.3f}"
    )
    print(f"Sampled choice-set size: {cfg.N_SAMPLED_ALTERNATIVES + 1}")
    print(f"Empty-set redraws: {data['empty_set_redraws']}")

    preview = np.column_stack(
        [
            data["consumer_id"],
            data["market_id"],
            data["chosen_product"],
            data["true_set"].sum(axis=1),
        ]
    )

    print("\nFirst 10 decisions:")
    print("consumer  market  chosen_product  true_set_size")
    print(preview[:10])
    print("\nAll simulation checks passed.")