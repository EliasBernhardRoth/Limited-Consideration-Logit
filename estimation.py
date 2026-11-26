import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

import config as cfg
from simulation import simulate_data


MODEL_LABELS = {
    "full": "Full-information logit (FILM)",
    "sampled": "Sampled-alternatives logit (MLM)",
    "true": "True-choice-set logit (WLM)",
}


def prepare_data(data, choice_set):
    """Validate prices, observed choices and the selected choice-set mask."""
    if choice_set not in MODEL_LABELS:
        raise ValueError("choice_set must be 'full', 'sampled' or 'true'.")

    prices = np.asarray(data["prices"], dtype=float)
    chosen = np.asarray(data["chosen_product"], dtype=float)
    available = np.asarray(data[f"{choice_set}_set"])

    if prices.ndim != 2 or prices.shape[0] == 0 or prices.shape[1] < 2:
        raise ValueError(
            "prices must be a nonempty decision-by-product matrix."
        )

    n_decisions, n_products = prices.shape

    if not np.all(np.isfinite(prices)):
        raise ValueError("Prices must be finite.")

    if (
        chosen.shape != (n_decisions,)
        or not np.all(np.isfinite(chosen))
        or np.any(chosen != np.floor(chosen))
        or np.any(chosen < 1)
        or np.any(chosen > n_products)
    ):
        raise ValueError("Chosen products must be valid integer product IDs.")

    if (
        available.shape != prices.shape
        or not np.all(np.isin(available, [0, 1]))
    ):
        raise ValueError(
            "The choice-set mask must match prices and contain only 0/1."
        )

    available = available.astype(bool)
    chosen_indices = chosen.astype(np.int64) - 1

    if not np.all(available[np.arange(n_decisions), chosen_indices]):
        raise ValueError("Each chosen product must belong to its choice set.")

    if np.any(np.bincount(chosen_indices, minlength=n_products) == 0):
        raise ValueError(
            "Every product must be chosen at least once to estimate "
            "all product effects with finite coefficients. Use a larger sample."
        )

    return prices, chosen_indices, available


def log_choice_probabilities(parameters, prices, available):
    """Logit log probabilities with product 1 as the reference category."""
    utilities = prices * parameters[0]
    utilities[:, 1:] += parameters[1:]

    utilities = np.where(available, utilities, -np.inf)

    return utilities - logsumexp(
        utilities,
        axis=1,
        keepdims=True,
    )


def negative_log_likelihood(parameters, prices, chosen_indices, available):
    """Return the total negative log likelihood and its analytical gradient."""
    log_probabilities = log_choice_probabilities(
        parameters, prices, available
    )
    rows = np.arange(len(prices))

    nll = -np.sum(log_probabilities[rows, chosen_indices])

    residuals = np.exp(log_probabilities)
    residuals[rows, chosen_indices] -= 1.0

    gradient = np.empty(prices.shape[1])
    gradient[0] = np.sum(residuals * prices)
    gradient[1:] = residuals[:, 1:].sum(axis=0)

    return float(nll), gradient


def likelihood_hessian(parameters, prices, available):
    """Return the Hessian of the total negative log likelihood."""
    probabilities = np.exp(
        log_choice_probabilities(parameters, prices, available)
    )

    expected_prices = np.sum(probabilities * prices, axis=1)
    centered_prices = prices - expected_prices[:, None]
    product_probabilities = probabilities[:, 1:]

    n_parameters = prices.shape[1]
    hessian = np.empty((n_parameters, n_parameters))

    hessian[0, 0] = np.sum(
        probabilities * centered_prices**2
    )

    price_product_terms = np.sum(
        product_probabilities * centered_prices[:, 1:],
        axis=0,
    )
    hessian[0, 1:] = price_product_terms
    hessian[1:, 0] = price_product_terms

    hessian[1:, 1:] = (
        np.diag(product_probabilities.sum(axis=0))
        - product_probabilities.T @ product_probabilities
    )

    return hessian


def estimate_model(data, choice_set):
    """
    Estimate price sensitivity and product effects for one choice-set model.

    Standard errors use the inverse observed information, as in the
    original R workflow. These are within-sample, model-based standard errors.
    """
    prices, chosen_indices, available = prepare_data(data, choice_set)
    n_decisions, n_products = prices.shape

    initial_parameters = np.zeros(n_products)

    # Scaling by the number of decisions improves numerical optimization.
    def objective(parameters):
        nll, gradient = negative_log_likelihood(
            parameters,
            prices,
            chosen_indices,
            available,
        )
        return nll / n_decisions, gradient / n_decisions

    def curvature(parameters):
        return (
            likelihood_hessian(parameters, prices, available)
            / n_decisions
        )

    initial_nll = objective(initial_parameters)[0] * n_decisions

    optimizer = "trust-exact"
    result = minimize(
        objective,
        initial_parameters,
        method=optimizer,
        jac=True,
        hess=curvature,
        options={"gtol": 1e-8, "maxiter": 1000},
    )

    if not result.success or not np.isfinite(result.fun):
        optimizer = "BFGS"
        starting_values = (
            result.x
            if np.all(np.isfinite(result.x))
            else initial_parameters
        )

        result = minimize(
            objective,
            starting_values,
            method=optimizer,
            jac=True,
            options={"gtol": 1e-8, "maxiter": 1000},
        )

    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(
            f"{MODEL_LABELS[choice_set]} did not converge: "
            f"{result.message}"
        )

    final_nll, gradient = negative_log_likelihood(
        result.x,
        prices,
        chosen_indices,
        available,
    )

    if final_nll > initial_nll + 1e-6:
        raise RuntimeError(
            "The fitted likelihood is worse than the starting likelihood."
        )

    # Use the TOTAL information matrix for standard errors.
    information = likelihood_hessian(
        result.x,
        prices,
        available,
    )

    if (
        not np.all(np.isfinite(information))
        or np.linalg.cond(information) > 1e12
    ):
        raise RuntimeError(
            "The information matrix is singular or poorly conditioned."
        )

    try:
        factor = np.linalg.cholesky(information)
        inverse_factor = np.linalg.solve(
            factor,
            np.eye(n_products),
        )
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "The information matrix is not positive definite."
        ) from error

    covariance = inverse_factor.T @ inverse_factor
    standard_errors = np.sqrt(np.diag(covariance))

    return {
        "model": choice_set,
        "coefficient_names": ["price"] + [
            f"product_{product}"
            for product in range(2, n_products + 1)
        ],
        "coefficients": result.x.copy(),
        "standard_errors": standard_errors,
        "covariance": covariance,
        "negative_log_likelihood": float(final_nll),
        "initial_negative_log_likelihood": float(initial_nll),
        "log_likelihood": -float(final_nll),
        "n_decisions": n_decisions,
        "converged": bool(result.success),
        "optimizer": optimizer,
        "iterations": int(result.nit),
        "max_average_gradient": float(
            np.max(np.abs(gradient / n_decisions))
        ),
        "message": str(result.message),
    }


if __name__ == "__main__":
    experiment = cfg.DEMO_EXPERIMENT

    data = simulate_data(
        n_consumers=experiment["n_consumers"],
        n_markets=experiment["n_markets"],
        seed=cfg.simulation_seed(1),
    )

    true_parameters = np.asarray(cfg.TRUE_COEFFICIENTS)

    for choice_set in MODEL_LABELS:
        fit = estimate_model(data, choice_set)

        print(f"\n{MODEL_LABELS[choice_set]}")
        print(
            f"{'Parameter':<14}"
            f"{'True':>12}"
            f"{'Estimate':>12}"
            f"{'Std. error':>12}"
        )

        for name, truth, estimate, se in zip(
            fit["coefficient_names"],
            true_parameters,
            fit["coefficients"],
            fit["standard_errors"],
        ):
            print(
                f"{name:<14}"
                f"{truth:>12.6f}"
                f"{estimate:>12.6f}"
                f"{se:>12.6f}"
            )

        print(
            f"Negative log likelihood: "
            f"{fit['negative_log_likelihood']:.6f}"
        )
        print(f"Converged: {fit['converged']}")
        print(f"Optimizer: {fit['optimizer']}")
        print(
            f"Maximum average gradient: "
            f"{fit['max_average_gradient']:.2e}"
        )

    print("\nAll three models estimated successfully.")