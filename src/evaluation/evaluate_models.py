
import pandas as pd


def combine_model_results(model_results):
    """
    Combine evaluation results from different models.

    Parameters:
        model_results: Dictionary containing model names
                       and their evaluation DataFrames.

    Returns:
        Combined DataFrame containing all model metrics.
    """

    combined_results = []

    required_columns = {
        "City",
        "MAE",
        "RMSE",
        "R2"
    }

    for model_name, results_df in model_results.items():

        if not isinstance(results_df, pd.DataFrame):
            raise TypeError(
                f"{model_name} results must be a DataFrame."
            )

        missing_columns = required_columns - set(results_df.columns)

        if missing_columns:
            raise ValueError(
                f"{model_name} is missing columns: "
                f"{missing_columns}"
            )

        model_df = results_df.copy()

        model_df["Model"] = model_name

        combined_results.append(model_df)

    if not combined_results:
        raise ValueError("No model results were provided.")

    final_results = pd.concat(
        combined_results,
        ignore_index=True
    )

    return final_results


def get_best_model(combined_results, metric="MAE"):
    """
    Identify the best-performing model for each city.

    MAE and RMSE: lower is better.
    R2: higher is better.
    """

    if metric not in ["MAE", "RMSE", "R2"]:
        raise ValueError(
            "Metric must be MAE, RMSE, or R2."
        )

    ascending = metric != "R2"

    valid_results = combined_results.dropna(
        subset=[metric]
    )

    best_models = (
        valid_results
        .sort_values(metric, ascending=ascending)
        .groupby("City", as_index=False)
        .first()
    )

    return best_models[
        ["City", "Model", "MAE", "RMSE", "R2"]
    ]


def create_comparison_table(combined_results):
    """
    Create a comparison table for all models and cities.
    """

    comparison_table = combined_results.pivot(
        index="City",
        columns="Model",
        values="MAE"
    )

    return comparison_table
