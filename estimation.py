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
    """Prepare prices, observed choices and choice sets."""

    if choice_set not in MODEL_LABELS:
        raise ValueError("Invalid choice-set model.")

    prices = np.asarray(data["prices"], dtype=float)
    chosen = np.asarray(data["chosen_product"], dtype=float)
    available = np.asarray(data[f"{choice_set}_set"], dtype=bool)

    n_decisions, n_products = prices.shape

    if not np.all(np.isfinite(prices)):
        raise ValueError("Prices must be finite.")

    if not np.all((chosen >= 1) & (chosen <= n_products)):
        raise ValueError("Invalid product IDs.")

    if np.any(chosen != np.floor(chosen)):
        raise ValueError("Product IDs must be integers.")

    # Convert product IDs to Python indices.
    chosen_indices = chosen.astype(np.int64) - 1

    rows = np.arange(n_decisions)

    if not np.all(available[rows, chosen_indices]):
        raise ValueError("Each chosen product must belong to its choice set.")

    # Every product must be chosen at least once.
    for product in range(n_products):
        if not np.any(chosen_indices == product):
            raise ValueError(
                "Every product must be chosen at least once. "
                "Use a larger sample."
            )

    return prices, chosen_indices, available


def log_choice_probabilities(parameters, prices, available):
    """Calculate logit log probabilities."""

    # Product 1 is the reference category.
    utilities = prices * parameters[0]
    utilities[:, 1:] += parameters[1:]

    # Exclude products outside the choice set.
    utilities[~available] = -np.inf

    return utilities - logsumexp(
        utilities,
        axis=1,
        keepdims=True,
    )


def negative_log_likelihood(
    parameters, prices, chosen_indices, available
):
    """Calculate the negative log likelihood and its gradient."""

    log_probabilities = log_choice_probabilities(
        parameters, prices, available
    )

    rows = np.arange(len(prices))

    # Sum log probabilities of observed choices.
    nll = -np.sum(log_probabilities[rows, chosen_indices])

    # Difference between predicted and observed choices.
    residuals = np.exp(log_probabilities)
    residuals[rows, chosen_indices] -= 1.0

    gradient = np.empty(prices.shape[1])

    gradient[0] = np.sum(residuals * prices)
    gradient[1:] = residuals[:, 1:].sum(axis=0)

    return float(nll), gradient


def likelihood_hessian(parameters, prices, available):
    """Calculate the Hessian of the negative log likelihood."""

    log_probabilities = log_choice_probabilities(
        parameters, prices, available
    )

    probabilities = np.exp(log_probabilities)

    expected_prices = np.sum(probabilities * prices, axis=1)
    centered_prices = prices - expected_prices[:, None]

    product_probabilities = probabilities[:, 1:]

    n_parameters = prices.shape[1]
    hessian = np.empty((n_parameters, n_parameters))

    # Second derivative with respect to the price coefficient.
    hessian[0, 0] = np.sum(
        probabilities * centered_prices**2
    )

    # Cross derivatives between price and product coefficients.
    price_product_terms = np.sum(
        product_probabilities * centered_prices[:, 1:],
        axis=0,
    )

    hessian[0, 1:] = price_product_terms
    hessian[1:, 0] = price_product_terms

    # Second derivatives of the product coefficients.
    hessian[1:, 1:] = (
        np.diag(product_probabilities.sum(axis=0))
        - product_probabilities.T @ product_probabilities
    )

    return hessian


def estimate_model(data, choice_set):
    """
    Estimate price sensitivity and product effects.

    Model-based standard errors are calculated using
    the inverse observed information matrix.
    """

    prices, chosen_indices, available = prepare_data(data, choice_set)

    n_decisions, n_products = prices.shape

    initial_parameters = np.zeros(n_products)

    # Scale the objective by the number of decisions.
    def objective(parameters):

        nll, gradient = negative_log_likelihood(
            parameters,
            prices,
            chosen_indices,
            available,
        )

        return nll / n_decisions, gradient / n_decisions

    def curvature(parameters):

        hessian = likelihood_hessian(
            parameters, prices, available
        )

        return hessian / n_decisions

    initial_nll = objective(initial_parameters)[0] * n_decisions

    # First optimisation attempt.
    optimizer = "trust-exact"

    result = minimize(
        objective,
        initial_parameters,
        method=optimizer,
        jac=True,
        hess=curvature,
        options={
            "gtol": 1e-8,
            "maxiter": 1000,
        },
    )

    # Use BFGS if the first optimisation fails.
    if not result.success or not np.isfinite(result.fun):

        optimizer = "BFGS"

        if np.all(np.isfinite(result.x)):
            starting_values = result.x
        else:
            starting_values = initial_parameters

        result = minimize(
            objective,
            starting_values,
            method=optimizer,
            jac=True,
            options={
                "gtol": 1e-8,
                "maxiter": 1000,
            },
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

    # Calculate the information matrix using the total likelihood.
    information = likelihood_hessian(
        result.x,
        prices,
        available,
    )

    if not np.all(np.isfinite(information)):
        raise RuntimeError("Non-finite information matrix.")

    if np.linalg.cond(information) > 1e12:
        raise RuntimeError("Poorly conditioned information matrix.")

    # Calculate the covariance matrix using Cholesky decomposition.
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

    # Name the estimated coefficients.
    coefficient_names = ["price"]

    for product in range(2, n_products + 1):
        coefficient_names.append(f"product_{product}")

    max_average_gradient = np.max(
        np.abs(gradient / n_decisions)
    )

    return {
        "model": choice_set,
        "coefficient_names": coefficient_names,
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
        "max_average_gradient": float(max_average_gradient),
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