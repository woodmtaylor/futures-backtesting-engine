import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis."""
    MIN_SAMPLES = 30
    WIN_MFE_THRESHOLD = 25
    SIGNIFICANCE_LEVEL = 0.05
    TICK_VALUE = 50  # ES contract tick value in dollars
    WIN_TICK_TARGET = 25  # Target ticks for win calculation
    STOP_TICK_LOSS = 9  # Stop loss in ticks


class RegressionAnalyzer:
    """Handles regression analysis and interaction testing."""

    @staticmethod
    def run_interaction_regression(data: pd.DataFrame, y_var: str, x_vars: List[str]) -> Tuple[float, float]:
        """
        Run regression analysis for interaction between two variables.

        Args:
            data: DataFrame with signal data
            y_var: Dependent variable (forward return)
            x_vars: List of two independent variables for interaction

        Returns:
            p_value and coefficient for the interaction term
        """
        if len(x_vars) != 2:
            raise ValueError("Exactly two variables required for interaction analysis")

        data_clean = data.copy()
        data_clean['Interaction'] = data_clean[x_vars[0]] * data_clean[x_vars[1]]

        X = data_clean[['Interaction']].copy()
        X = sm.add_constant(X)
        y = data_clean[y_var]

        model = sm.OLS(y, X, missing='drop').fit()

        return model.pvalues['Interaction'], model.params['Interaction']


class DistributionAnalyzer:
    """Analyzes return distributions and calculates performance metrics."""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def analyze_returns(self, data: pd.DataFrame, column: str) -> Tuple[int, int, float]:
        """
        Analyze distribution of returns.

        Returns:
            Number above zero, number below zero, mean value
        """
        above_zero = (data[column] > 0).sum()
        below_zero = (data[column] < 0).sum()
        mean_value = data[column].mean()

        return above_zero, below_zero, mean_value

    def calculate_performance_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate trading performance metrics."""
        num_wins = (data['MFE'] >= self.config.WIN_MFE_THRESHOLD).sum()
        num_stops = len(data) - num_wins
        total_trades = len(data)

        if total_trades == 0:
            return {'wins': 0, 'stops': 0, 'winrate': 0, 'pnl': 0}

        winrate = num_wins / total_trades
        win_value = self.config.WIN_TICK_TARGET / 4 * self.config.TICK_VALUE
        stop_value = self.config.STOP_TICK_LOSS / 4 * self.config.TICK_VALUE
        pnl = (num_wins * win_value) - (num_stops * stop_value)

        return {
            'wins': num_wins,
            'stops': num_stops,
            'winrate': winrate,
            'pnl': pnl
        }


class OptimizationEngine:
    """Finds optimal thresholds for trading signals."""

    @staticmethod
    def find_optimal_threshold(data: pd.DataFrame, interaction_col: str,
                               forward_return_col: str, min_samples: int,
                               num_bins: int = 100) -> Tuple[Optional[float], float]:
        """
        Find optimal interaction value that maximizes mean forward returns.

        Args:
            data: DataFrame with interaction and return data
            interaction_col: Column name for interaction values
            forward_return_col: Column name for forward returns
            min_samples: Minimum sample size required
            num_bins: Number of percentile bins to test

        Returns:
            Optimal threshold value and maximum mean return
        """
        data_clean = data.copy()
        data_clean[interaction_col] = pd.to_numeric(data_clean[interaction_col], errors='coerce')

        # Generate threshold candidates using percentiles
        percentiles = np.linspace(0, 100, num_bins + 1)
        threshold_candidates = np.percentile(data_clean[interaction_col], percentiles)

        optimal_threshold = None
        max_mean_return = float('-inf')

        for threshold in threshold_candidates:
            filtered_data = data_clean[data_clean[interaction_col] >= threshold]

            if len(filtered_data) >= min_samples:
                mean_return = filtered_data[forward_return_col].mean()

                if mean_return > max_mean_return:
                    max_mean_return = mean_return
                    optimal_threshold = threshold

        return optimal_threshold, max_mean_return


class Visualizer:
    """Handles plotting and visualization of analysis results."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.dist_analyzer = DistributionAnalyzer(config)

    def plot_return_distribution(self, data: pd.DataFrame, column: str, title: str,
                                 col1: str, col2: str, dep_var: str) -> None:
        """
        Plot return distribution with performance metrics.

        Args:
            data: Filtered data for plotting
            column: Column to plot (forward returns)
            title: Plot title
            col1, col2: Signal column names for context
            dep_var: Dependent variable name
        """
        try:
            # Validate data
            if not self._validate_plot_data(data, dep_var):
                return

            # Calculate performance metrics
            metrics = self.dist_analyzer.calculate_performance_metrics(data)

            # Check minimum requirements
            if len(data) < self.config.MIN_SAMPLES or metrics['winrate'] <= 0.25:
                return

            # Create plot
            self._create_distribution_plot(data, column, title, metrics)

        except Exception as e:
            print(f"Skipping plot for {title} due to error: {e}")

    def _validate_plot_data(self, data: pd.DataFrame, dep_var: str) -> bool:
        """Validate data requirements for plotting."""
        stop_col = f'stop_{dep_var}'
        if stop_col not in data.columns:
            print(f"Stop column {stop_col} not found in data.")
            return False
        return True

    def _create_distribution_plot(self, data: pd.DataFrame, column: str,
                                  title: str, metrics: Dict[str, float]) -> None:
        """Create the actual distribution plot with metrics using broken axis if needed."""
        # Check if we need a broken axis (high frequency near zero with sparse outliers)
        hist_values, bin_edges = np.histogram(data[column], bins=30)
        max_freq = hist_values.max()

        # Check for outlier bins (much lower frequency than main distribution)
        outlier_threshold = max_freq * 0.1  # 10% of peak frequency
        main_distribution_range = self._find_main_distribution_range(data[column], hist_values, bin_edges)

        # Use broken axis if we have sparse outliers beyond the main distribution
        if self._should_use_broken_axis(data[column], main_distribution_range, outlier_threshold):
            self._create_broken_y_axis_plot(data, column, title, metrics, main_distribution_range)
        else:
            self._create_standard_plot(data, column, title, metrics)

    def _find_main_distribution_range(self, data: pd.Series, hist_values: np.ndarray,
                                      bin_edges: np.ndarray) -> Tuple[float, float]:
        """Find the range that contains the main distribution."""
        # Find bins with significant frequency (>5% of max)
        significant_bins = hist_values > (hist_values.max() * 0.05)
        significant_indices = np.where(significant_bins)[0]

        if len(significant_indices) == 0:
            return data.min(), data.max()

        # Get the range of significant bins
        start_idx = significant_indices[0]
        end_idx = significant_indices[-1] + 1  # +1 because bin_edges has one more element

        return bin_edges[start_idx], bin_edges[end_idx]

    def _should_use_broken_axis(self, data: pd.Series, main_range: Tuple[float, float],
                                outlier_threshold: float) -> bool:
        """Determine if broken axis would improve visualization."""
        main_min, main_max = main_range

        # Check if there are outliers beyond the main range
        outliers_beyond = data[(data < main_min) | (data > main_max)]
        main_data = data[(data >= main_min) & (data <= main_max)]

        # Use broken axis if:
        # 1. We have outliers beyond main range
        # 2. Main distribution contains >70% of data
        # 3. Range of outliers is significant compared to main range
        if len(outliers_beyond) > 0 and len(main_data) > len(data) * 0.7:
            main_span = main_max - main_min
            total_span = data.max() - data.min()
            if total_span > main_span * 2:  # Significant outlier range
                return True

        return False

    def _create_standard_plot(self, data: pd.DataFrame, column: str,
                              title: str, metrics: Dict[str, float]) -> None:
        """Create standard single-axis plot."""
        plt.figure(figsize=(8, 6))

        # Main histogram and KDE
        sns.histplot(data[column], bins=30, kde=True)

        # Reference lines
        plt.axvline(x=0, color='red', linestyle='--', label='Zero Line')
        plt.axvline(x=data[column].mean(), color='green', linestyle='--', label='Mean')

        # Add performance metrics text
        self._add_metrics_text(data[column].mean(), metrics, len(data))

        # Format plot
        plt.title(f"{title}\nTotal Observations: {len(data)}", pad=20)
        plt.xlabel('Forward Return (ticks)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _create_broken_y_axis_plot(self, data: pd.DataFrame, column: str, title: str,
                                   metrics: Dict[str, float], main_range: Tuple[float, float]) -> None:
        """Create a broken Y-axis plot with break point based on second-highest bar."""
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8),
                                       gridspec_kw={'height_ratios': [1, 4]})  # Small top, large bottom
        fig.subplots_adjust(hspace=0.05)

        # Calculate histogram for both plots
        hist_values, bin_edges = np.histogram(data[column], bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        # Find the second-highest frequency value
        sorted_frequencies = np.sort(hist_values)
        second_highest_freq = sorted_frequencies[-2] if len(sorted_frequencies) > 1 else sorted_frequencies[-1]
        max_freq = hist_values.max()

        # Set break point slightly above the second-highest bar
        break_point = second_highest_freq * 1.1  # 10% above second-highest bar

        # Plot the same data on both axes
        ax1.bar(bin_centers, hist_values, width=bin_width, alpha=0.7, edgecolor='black')
        ax2.bar(bin_centers, hist_values, width=bin_width, alpha=0.7, edgecolor='black')

        # Set different y-limits for each subplot
        ax1.set_ylim(break_point, max_freq * 1.02)  # Top plot: just the tallest bar(s)
        ax2.set_ylim(0, break_point)  # Bottom plot: everything else with full detail

        # Add reference lines to both plots
        for ax in [ax1, ax2]:
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Zero Line')
            ax.axvline(x=data[column].mean(), color='green', linestyle='--', alpha=0.8, linewidth=2, label='Mean')

        # Hide spines between the plots
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        # Remove x-axis ticks from top plot
        ax1.tick_params(bottom=False, labelbottom=False)

        # Add break indicators
        d = 0.3  # Size of diagonal lines
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                      linestyle="none", color='k', mec='k', mew=1.5, clip_on=False)
        ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

        # Add performance metrics to the top plot (positioned carefully due to small space)
        self._add_metrics_text_broken_y_compact(ax1, data[column].mean(), metrics, len(data))

        # Add legend to bottom plot
        ax2.legend(loc='upper right')

        # Labels with better positioning
        ax2.set_xlabel('Forward Return (ticks)', fontsize=12)
        fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=12)  # Moved further left

        # Title with more top margin
        fig.suptitle(f"{title}\nTotal Observations: {len(data)}", fontsize=14, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, left=0.1)  # Add space at top and left
        plt.show()

    def _add_metrics_text_broken_y_compact(self, ax, mean_value: float, metrics: Dict[str, float],
                                           total_obs: int) -> None:
        """Add performance metrics text to small top subplot with vertical layout."""
        text_y_positions = [0.95, 0.75, 0.55, 0.35, 0.15]

        text_items = [
            f'Mean: {mean_value:.2f}',
            f'Stops: {metrics["stops"]:.0f}',
            f'Wins: {metrics["wins"]:.0f}',
            f'Winrate: {metrics["winrate"]:.2%}',
            f'PnL: ${metrics["pnl"]:.2f}'
        ]

        for i, text in enumerate(text_items):
            ax.text(0.98, text_y_positions[i], text,
                    transform=ax.transAxes, fontsize=9,  # Smaller font size
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))

    def _add_metrics_text_broken_y(self, ax, mean_value: float, metrics: Dict[str, float],
                                   total_obs: int) -> None:
        """Add performance metrics text to broken Y-axis plot."""
        text_y_positions = [0.95, 0.85, 0.75, 0.65, 0.55]

        text_items = [
            f'Mean: {mean_value:.2f}',
            f'Stops: {metrics["stops"]:.0f}',
            f'Wins: {metrics["wins"]:.0f}',
            f'Winrate: {metrics["winrate"]:.2%}',
            f'PnL: ${metrics["pnl"]:.2f}'
        ]

        for i, text in enumerate(text_items):
            ax.text(0.98, text_y_positions[i], text,
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

    def _add_metrics_text(self, mean_value: float, metrics: Dict[str, float],
                          total_obs: int) -> None:
        """Add performance metrics text to plot."""
        text_y_positions = [0.85, 0.80, 0.75, 0.70, 0.65]

        text_items = [
            f'Mean: {mean_value:.2f}',
            f'Stops: {metrics["stops"]:.0f}',
            f'Wins: {metrics["wins"]:.0f}',
            f'Winrate: {metrics["winrate"]:.2%}',
            f'PnL: ${metrics["pnl"]:.2f}'
        ]

        for i, text in enumerate(text_items):
            plt.text(0.95, text_y_positions[i], text,
                     transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right')


class SignalAnalysisEngine:
    """Main engine for comprehensive signal analysis."""

    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
        self.regression_analyzer = RegressionAnalyzer()
        self.dist_analyzer = DistributionAnalyzer(self.config)
        self.optimizer = OptimizationEngine()
        self.visualizer = Visualizer(self.config)

    def load_and_prepare_data(self, file_path: str) -> pd.DataFrame:
        """Load and prepare data for analysis."""
        data = pd.read_csv(file_path)

        # Filter to inversion signals only
        data = data[data['Inversion'] == 1].copy()

        # Calculate TD% metric
        data['TD%'] = np.where(
            data['Buy INV'] == 1,
            data['Close'] / data['Low.2'],
            np.where(
                data['Sell INV'] == 1,
                data['Close'] / data['High.1'],
                np.nan
            )
        )

        return data

    def run_comprehensive_analysis(self, data: pd.DataFrame) -> Dict[str, List]:
        """
        Run comprehensive interaction analysis across all signal combinations.

        Returns:
            Dictionary of significant interactions by dependent variable
        """
        # Define analysis columns
        dependent_vars = ['diff_5', 'diff_10', 'diff_20', 'diff_30',
                          'diff_40', 'diff_50', 'diff_75', 'diff_100']

        inversion_columns = ['Buy INV Strong', 'Buy INV Weak', 'Sell INV Strong',
                             'Sell INV Weak', 'Buy INV', 'Sell INV',
                             'Strong Inversion', 'Weak Inversion']

        exhaustion_columns = ['Exhaustion', 'Strong EX', 'Weak EX', 'Buy EX',
                              'Sell EX', 'Buy EX Strong', 'Buy EX Weak',
                              'Sell EX Strong', 'Sell EX Weak']

        other_columns = ['Vol*', 'Vol Outlier', 'Delta %', 'TD%']

        # Store results
        significant_interactions = {dep_var: [] for dep_var in dependent_vars}
        plot_data = []

        # Test all combinations
        for dep_var in dependent_vars:
            for col1 in inversion_columns:
                for col2 in (exhaustion_columns + other_columns):
                    interaction_result = self._analyze_interaction(
                        data, dep_var, col1, col2
                    )

                    if interaction_result:
                        significant_interactions[dep_var].append(interaction_result)

                        # Prepare plot data if meets criteria
                        plot_info = self._prepare_plot_data(
                            data, interaction_result, dep_var, col1, col2
                        )
                        if plot_info:
                            plot_data.append(plot_info)

        # Generate plots
        self._generate_plots(plot_data)

        return significant_interactions

    def _analyze_interaction(self, data: pd.DataFrame, dep_var: str,
                             col1: str, col2: str) -> Optional[Tuple]:
        """Analyze single interaction pair."""
        try:
            p_value, coefficient = self.regression_analyzer.run_interaction_regression(
                data, dep_var, [col1, col2]
            )

            if p_value > self.config.SIGNIFICANCE_LEVEL:
                return None

            # Create interaction data
            interaction_data = self._create_interaction_dataset(
                data, col1, col2, dep_var
            )

            # Find optimal threshold
            optimal_threshold, max_mean = self.optimizer.find_optimal_threshold(
                interaction_data, 'Interaction', 'Forward_Return',
                self.config.MIN_SAMPLES
            )

            if optimal_threshold is None or optimal_threshold == 0:
                return None

            # Calculate mean return for filtered data
            filtered_data = interaction_data[
                abs(interaction_data['Interaction']) >= abs(optimal_threshold)
                ]
            mean_return = filtered_data['Forward_Return'].mean()

            return (col1, col2, p_value, coefficient, mean_return, optimal_threshold)

        except Exception as e:
            print(f"Error analyzing {col1} x {col2} for {dep_var}: {e}")
            return None

    def _create_interaction_dataset(self, data: pd.DataFrame, col1: str,
                                    col2: str, dep_var: str) -> pd.DataFrame:
        """Create dataset with interaction terms and required columns."""
        return pd.DataFrame({
            'I1': col1,
            'I2': col2,
            'Interaction': data[col1] * data[col2],
            'MFE': data['MFE'],
            'Forward_Return': data[dep_var],
            **{f'stop_{period}': data[f'stop_{period}']
               for period in ['diff_5', 'diff_10', 'diff_20', 'diff_30',
                              'diff_40', 'diff_50', 'diff_75', 'diff_100']}
        })

    def _prepare_plot_data(self, data: pd.DataFrame, interaction_result: Tuple,
                           dep_var: str, col1: str, col2: str) -> Optional[Tuple]:
        """Prepare data for plotting if it meets minimum criteria."""
        _, _, _, _, _, optimal_threshold = interaction_result

        interaction_data = self._create_interaction_dataset(data, col1, col2, dep_var)
        filtered_data = interaction_data[
            abs(interaction_data['Interaction']) >= abs(optimal_threshold)
            ]
        filtered_data = filtered_data[filtered_data['Forward_Return'].notna()]

        if len(filtered_data) >= self.config.MIN_SAMPLES:
            title = (f"Forward Return Distribution - {col1} × {col2} ({dep_var})\n"
                     f"Optimal Interaction Value: {optimal_threshold:.2f}")

            return (filtered_data, 'Forward_Return', title, col1, col2, dep_var)

        return None

    def _generate_plots(self, plot_data: List[Tuple]) -> None:
        """Generate all distribution plots."""
        for data_tuple in plot_data:
            filtered_data, column, title, col1, col2, dep_var = data_tuple
            self.visualizer.plot_return_distribution(
                filtered_data, column, title, col1, col2, dep_var
            )

    def print_results(self, significant_interactions: Dict[str, List]) -> None:
        """Print formatted results of significant interactions."""
        for dep_var, interactions in significant_interactions.items():
            if not interactions:
                continue

            print(f"\nSignificant interactions for {dep_var}:")
            print("-" * 100)
            print(f"{'Column 1':<20} {'Column 2':<20} {'P-value':<10} "
                  f"{'Coefficient':<15} {'Mean Return':<12} {'Optimal Value':<15}")
            print("-" * 100)

            for interaction in interactions:
                col1, col2, p_val, coeff, mean_ret, opt_val = interaction
                print(f"{col1:<20} {col2:<20} {p_val:<10.4f} "
                      f"{coeff:<15.4f} {mean_ret:<12.4f} {opt_val:<15.2f}")


def main():
    """Main execution function."""
    # Initialize analysis engine
    config = AnalysisConfig()
    engine = SignalAnalysisEngine(config)

    # Load and prepare data
    data = engine.load_and_prepare_data('8_tick_inv_and_ex.csv')

    # Run comprehensive analysis
    results = engine.run_comprehensive_analysis(data)

    # Print results
    engine.print_results(results)


if __name__ == "__main__":
    main()