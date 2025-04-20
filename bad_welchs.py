import pandas as pd
from scipy.stats import f as f_dist

# Assuming your data is in a pandas DataFrame called 'data'

def welchs_anova_simplified(data, dv, between):
    """
    Performs Welch's ANOVA for three independent groups.

    Args:
        data (pd.DataFrame): The dataframe containing the data.
        dv (str): The name of the dependent variable column.
        between (str): The name of the grouping variable column (assumed to have 3 unique values).

    Returns:
        pandas.DataFrame: A summary table of the Welch's ANOVA results.
    """
    groups = data[between].unique()
    group_data = {group: data[data[between] == group][dv] for group in groups}
    group_means = {group: group_data[group].mean() for group in groups}
    n_groups = len(groups)

    weights = {group: len(group_data[group]) / (group_data[group].var(ddof=1) if len(group_data[group]) > 1 and group_data[group].var(ddof=1) != 0 else 1e-9) for group in groups}
    weight_sum = sum(weights.values())
    grand_mean_welch = sum(weights[group] * group_means[group] for group in groups) / weight_sum

    f_statistic_numerator = sum(weights[group] * (group_means[group] - grand_mean_welch)**2 for group in groups)

    c = 1 + (2 / (n_groups**2 - 1)) * sum((1 - weights[group] / weight_sum)**2 / (len(group_data[group]) - 1) for group in groups if len(group_data[group]) > 1)

    f_statistic = f_statistic_numerator / c if c != 0 else 0

    degrees_of_freedom_numerator = n_groups - 1
    degrees_of_freedom_denominator_numerator = (weight_sum**2) / sum((weights[group]**2 / (len(group_data[group]) - 1)) for group in groups if len(group_data[group]) > 1)
    degrees_of_freedom_denominator = degrees_of_freedom_denominator_numerator - (n_groups - 1)

    p_value = f_dist.sf(f_statistic, degrees_of_freedom_numerator, degrees_of_freedom_denominator)

    results = pd.DataFrame({
        "F": [f_statistic],
        "p-value": [p_value],
        "ddof1": [degrees_of_freedom_numerator],
        "ddof2": [degrees_of_freedom_denominator]
    })
    return results

# Perform Welch's ANOVA on spending by store location
welch_anova_results = welchs_anova_simplified(data, dv="Amount_Spent", between="Store_Location")
print("\n--- Welch's ANOVA Results (Spending by Location) ---")
print(welch_anova_results)