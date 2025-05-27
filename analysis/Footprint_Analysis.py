import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

plt.imshow([[0]], cmap='gray'); plt.axis('off'); plt.show()

# Run regression for an interaction
def run_regression_interaction(data, y_var, x_vars):
    data['Interaction'] = data[x_vars[0]] * data[x_vars[1]]  # Create interaction term
    X = data[['Interaction']]  # Include only the interaction term
    X = sm.add_constant(X)
    y = data[y_var]
    model = sm.OLS(y, X, missing='drop').fit()
    return model.pvalues['Interaction'], model.params['Interaction']

# Analyze datapoints above and below zero for plots
def analyze_distribution(data, column):
    above_zero = (data[column] > 0).sum()
    below_zero = (data[column] < 0).sum()
    mean_value = data[column].mean()
    return above_zero, below_zero, mean_value

def plot_distribution(filtered_data, column, title, samples, col1, col2, dep_var):
    try:
        stop_dep_var = 'stop_' + dep_var  # Stop column name
        if stop_dep_var not in filtered_data.columns:
            print(f"Stop column {stop_dep_var} not found in data.")
            return

        num_wins = (filtered_data['MFE'] >= win_MFE).sum()
        num_stops = len(filtered_data) - num_wins

        total_trades = len(filtered_data)
        total_interactions = len(filtered_data)
        
        if total_trades < samples or num_wins / total_trades <= 0.25:
            return

        plt.figure(figsize=(8, 6))
        sns.histplot(filtered_data[column], bins=30, kde=True)
        plt.axvline(x=0, color='red', linestyle='--', label='Zero Line')
        plt.axvline(x=filtered_data[column].mean(), color='green', linestyle='--', label='Mean')

        if filtered_data[column].var() > 0:
            kde = sns.kdeplot(filtered_data[column])
            if kde and kde.lines:
                x_values = kde.lines[0].get_xdata()
                y_values = kde.lines[0].get_ydata()
                net_area = np.trapz(y_values, x_values)
        else:
            print(f"Skipping KDE plot for {title} due to zero variance.")

        mean_value = filtered_data[column].mean()
        plt.text(0.95, 0.85, f'Mean: {mean_value:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')
        plt.text(0.95, 0.80, f'Stops: {num_stops:.0f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')
        plt.text(0.95, 0.75, f'Wins: {num_wins:.0f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')
        plt.text(0.95, 0.70, f'Winrate%: {(num_wins / total_trades):.2%}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')
        plt.text(0.95, 0.65, f'PnL: ${( (num_wins * 25/4*50) - (num_stops * 9/4*50) ):.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')

        plt.title(f"{title}\nPlotted Interactions (>= Optimal Value): {total_interactions:.0f}", pad=20)
        plt.xlabel('Forward Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()  
        plt.show()
        
    except Exception as e:
        print(f"Skipping graph for {title} due to an error: {e}")

def find_optimal_interaction_value(data, interaction_col, forward_return_col, samples, num_bins=100):
    data[interaction_col] = pd.to_numeric(data[interaction_col], errors='coerce')
    # Get the range of interaction values using percentiles
    percentiles = np.linspace(0, 100, num_bins + 1)
    interaction_values = np.percentile(data[interaction_col], percentiles)
    
    optimal_interaction_value = None
    max_mean = float('-inf')
    
    for value in interaction_values:
        filtered_data = data[data[interaction_col] >= value]
        if len(filtered_data) >= samples:
            mean_value = filtered_data[forward_return_col].mean()
            if mean_value > max_mean:
                max_mean = mean_value
                optimal_interaction_value = value
    
    return optimal_interaction_value, max_mean

# Load your data
data_lvl = pd.read_csv('8_tick_inv_and_ex.csv')

data_lvl.drop(data_lvl[data_lvl['Inversion'] != 1].index, inplace=True)
data_lvl['TD%'] = np.where(data_lvl['Buy INV'] == 1, data_lvl['Close'] / data_lvl['Low.2'],
                            np.where(data_lvl['Sell INV'] == 1, data_lvl['Close'] / data_lvl['High.1'], np.nan))
# data_lvl.dropna(subset=['diff_5'], inplace=True)

samples = 30 # Minimum number of samples
win_MFE = 25

# Define columns
dependent_vars = ['diff_5', 'diff_10', 'diff_20', 'diff_30', 'diff_40', 'diff_50', 'diff_75', 'diff_100']
inversion_columns = ['Buy INV Strong', 'Buy INV Weak', 'Sell INV Strong', 'Sell INV Weak', 'Buy INV', 'Sell INV', 'Strong Inversion', 'Weak Inversion']
exhaustion_columns = ['Exhaustion', 'Strong EX', 'Weak EX', 'Buy EX', 'Sell EX', 'Buy EX Strong', 'Buy EX Weak', 'Sell EX Strong', 'Sell EX Weak']
other_columns = ['Vol*', 'Vol Outlier', 'Delta %', 'TD%']

significant_interactions = {dep_var: [] for dep_var in dependent_vars}
plots = []

# Interaction between Inversion and Other columns
for dep_var in dependent_vars:
    for col1 in inversion_columns:
        for col2 in (exhaustion_columns + other_columns):
            p_value, coefficient = run_regression_interaction(data_lvl, dep_var, [col1, col2])
            
            if p_value <= .05:  # Only process if p value is significant (< 0.05)
                data_interaction = pd.DataFrame({
                    'I1': col1, 
                    'I2': col2, 
                    'Interaction': data_lvl[col1] * data_lvl[col2], 
                    'MFE': data_lvl['MFE'], 
                    'Forward_Return': data_lvl[dep_var], 
                    'stop_diff_5': data_lvl['stop_diff_5'], 
                    'stop_diff_10': data_lvl['stop_diff_10'], 
                    'stop_diff_20': data_lvl['stop_diff_20'], 
                    'stop_diff_30': data_lvl['stop_diff_30'], 
                    'stop_diff_40': data_lvl['stop_diff_40'], 
                    'stop_diff_50': data_lvl['stop_diff_50'], 
                    'stop_diff_75': data_lvl['stop_diff_75'], 
                    'stop_diff_100': data_lvl['stop_diff_100']
                })                
                optimal_value, max_mean = find_optimal_interaction_value(data_interaction, 'Interaction', 'Forward_Return', samples)
                
                if optimal_value is not None and optimal_value != 0:
                    total_interactions = len(data_interaction['Interaction'])
                    filtered_data = data_interaction.loc[abs(data_interaction['Interaction']) >= abs(optimal_value)]
                    filtered_data = filtered_data.loc[filtered_data['Forward_Return'].notna()]
                    above_zero, below_zero, mean_value = analyze_distribution(filtered_data, 'Forward_Return')
                    
                    significant_interactions[dep_var].append((col1, col2, p_value, coefficient, mean_value, optimal_value))
                    
                    if len(filtered_data) >= samples:  # Filter out interactions with less than 5 datapoints
                        title = f"Forward Return Distribution - {col1} × {col2} ({dep_var})\nOptimal Interaction Value: {optimal_value:.2f}"
                        plots.append((filtered_data, 'Forward_Return', title, len(filtered_data), col1, col2, dep_var))

# Print the significant interactions for each forward return
for dep_var, interactions in significant_interactions.items():
    print(f"Significant interactions for {dep_var}:")
    print("{:<20} {:<20} {:<10} {:<15} {:<10} {:<20}".format("Inversion Column", "Other Column", "P-value", "Coefficient", "Mean", "Optimal Value"))
    print("-" * 100)
    
    for col1, col2, p_value, coefficient, mean_value, optimal_value in interactions:
        print("{:<20} {:<20} {:<10.4f} {:<15.4f} {:<10.4f} {:<20.2f}".format(col1, col2, p_value, coefficient, mean_value, optimal_value))
    
    print("\n")

# Plot the graphs in the order of the significant interactions
for filtered_data, column, title, samples, col1, col2, dep_var in plots:
    plot_distribution(filtered_data, column, title, samples, col1, col2, dep_var)