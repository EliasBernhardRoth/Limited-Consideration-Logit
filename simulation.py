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
    Product IDs start at 1.

    Empty consideration sets are either redrawn at the same
    prices or treated as an error.
    """

    if n_consumers < 1 or n_markets < 1:
        raise ValueError("Consumers and markets must be positive.")

    if empty_set_policy not in ("resample", "raise"):
        raise ValueError("Invalid empty-set policy.")

    n_products = cfg.N_PRODUCTS
    n_sampled = cfg.N_SAMPLED_ALTERNATIVES

    list_prices = np.asarray(cfg.LIST_PRICES, dtype=float)
    product_effects = np.asarray(cfg.PRODUCT_EFFECTS, dtype=float)
    consideration_intercepts = np.asarray(
        cfg.CONSIDERATION_INTERCEPTS, dtype=float
    )

    n_decisions = int(n_consumers) * int(n_markets)
    rng = np.random.RandomState(seed)

    # Initialise arrays.
    prices = np.empty((n_decisions, n_products))
    probabilities = np.zeros((n_decisions, n_products))

    true_set = np.zeros((n_decisions, n_products), dtype=bool)
    sampled_set = np.zeros((n_decisions, n_products), dtype=bool)

    chosen_product = np.empty(n_decisions, dtype=np.int64)

    product_indices = np.arange(n_products)
    empty_set_redraws = 0

    # Simulate each consumer-market decision.
    for decision in range(n_decisions):

        # Generate prices.
        current_prices = (
            list_prices * cfg.PRICE_SCALE
            + cfg.PRICE_RANDOM_WIDTH * rng.random_sample(n_products)
        )

        # Calculate consideration probabilities.
        attention = (
            cfg.CONSIDERATION_PRICE_COEFFICIENT * current_prices
            + consideration_intercepts
        )

        consideration_probabilities = expit(attention)

        # Draw the consumer's consideration set.
        considered = (
            rng.random_sample(n_products) < consideration_probabilities
        )

        attempts = 1

        # Redraw empty consideration sets.
        while not np.any(considered):

            if empty_set_policy == "raise":
                raise RuntimeError(
                    f"Empty consideration set at decision {decision + 1}, "
                    f"seed={seed}."
                )

            if attempts >= MAX_CONSIDERATION_DRAWS:
                raise RuntimeError("Could not draw a nonempty consideration set.")

            considered = (
                rng.random_sample(n_products) < consideration_probabilities
            )

            empty_set_redraws += 1
            attempts += 1

        # Calculate product utilities.
        utilities = (
            cfg.PRICE_COEFFICIENT * current_prices
            + product_effects
            + cfg.UTILITY_SHOCK_STD * rng.standard_normal(n_products)
        )

        # Calculate choice probabilities within the consideration set.
        considered_indices = product_indices[considered]
        considered_probabilities = softmax(utilities[considered])

        # Draw the chosen product.
        cumulative_probabilities = np.cumsum(considered_probabilities)
        cumulative_probabilities[-1] = 1.0

        draw = rng.random_sample()

        position = np.searchsorted(
            cumulative_probabilities,
            draw,
            side="right",
        )

        chosen_index = considered_indices[position]

        # Sample alternatives from all non-chosen products.
        other_products = product_indices[product_indices != chosen_index]

        sampled_others = rng.choice(
            other_products,
            size=n_sampled,
            replace=False,
        )

        # Store the simulated decision.
        prices[decision] = current_prices
        true_set[decision] = considered

        probabilities[decision, considered] = considered_probabilities

        chosen_product[decision] = chosen_index + 1

        sampled_set[decision, chosen_index] = True
        sampled_set[decision, sampled_others] = True

    # Create consumer and market IDs.
    consumer_ids = np.repeat(
        np.arange(1, n_consumers + 1), n_markets
    )

    market_ids = np.tile(
        np.arange(1, n_markets + 1), n_consumers
    )

    return {
        "decision_id": np.arange(1, n_decisions + 1),
        "consumer_id": consumer_ids,
        "market_id": market_ids,
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

    # Check that prices are finite.
    assert np.all(np.isfinite(data["prices"]))

    # Check that the chosen product belongs to both choice sets.
    assert np.all(data["true_set"][rows, chosen_indices])
    assert np.all(data["sampled_set"][rows, chosen_indices])

    # Check the size of the sampled choice set.
    assert np.all(
        data["sampled_set"].sum(axis=1)
        == cfg.N_SAMPLED_ALTERNATIVES + 1
    )

    # Check that choice probabilities sum to one.
    assert np.allclose(
        data["probabilities"].sum(axis=1),
        1.0,
        rtol=0,
        atol=1e-12,
    )

    # Products outside the consideration set cannot be chosen.
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

    preview = np.column_stack([
        data["consumer_id"],
        data["market_id"],
        data["chosen_product"],
        data["true_set"].sum(axis=1),
    ])

    print("\nFirst 10 decisions:")
    print("consumer  market  chosen_product  true_set_size")
    print(preview[:10])
    print("\nAll simulation checks passed.")