import json
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, norm, shapiro
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Apply Seaborn theme for nicer plots
# Seaborn theme and context for very large figure text
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={
    'font.size': 28,             # Base font size
    'axes.titlesize': 36,        # Title size
    'axes.labelsize': 32,        # Axis label size
    'xtick.labelsize': 30,       # X tick label size
    'ytick.labelsize': 30,       # Y tick label size
    'legend.fontsize': 30,       # Legend font size
    'legend.title_fontsize': 32, # Legend title font size
    'figure.dpi': 600            # High DPI for print
})


class G08AEvaluator:
    def __init__(self, result_dir, output_dir, num_experiments) -> None:
        self.result_dir = result_dir
        self.output_dir = output_dir
        self.num_experiments = num_experiments
        self.all_experiment_data = [] # List to store dataframes from each experiment
        self.summary_stats = defaultdict(lambda: np.nan) # Default to NaN for missing stats
        self.summary_stats["team1_members"] = set()
        self.summary_stats["team2_members"] = set()
        self.summary_stats["total_experiments"] = 0 # Initialize explicitly
        self.summary_stats["total_rounds_processed"] = 0
        self.summary_stats["team1_wins"] = 0
        self.summary_stats["team2_wins"] = 0
        self.summary_stats["ties"] = 0
        self.persona_performance = defaultdict(lambda: {'total_deviation': 0.0, 'count': 0}) # For persona stats
        self.correlation_data = {} # For correlation results
        self.max_rounds_observed = 0


        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(f"Created output directory: {self.output_dir}")

    def load_and_process_experiment(self, file_path):
        """Loads data, validates, and processes it into a pandas DataFrame."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load or parse JSON from {file_path}: {e}")
            return None

        # Validate required keys
        if not all(k in data for k in ['winners', 'biddings']):
            logging.error(f"Missing 'winners' or 'biddings' key in {file_path}. Skipping.")
            return None
        if not data['biddings']:
             logging.warning(f"Biddings data is empty in {file_path}. Skipping.")
             return None

        experiment_id = os.path.splitext(os.path.basename(file_path))[0]
        team1, team2 = self.parse_teams_from_filename(file_path)

        # --- Convert biddings to DataFrame ---
        records = []
        max_rounds_exp = 0
        for player, bids in data['biddings'].items():
            max_rounds_exp = max(max_rounds_exp, len(bids))
            team_id = 1 if player in team1 else (2 if player in team2 else 0)
            for round_num, bid in enumerate(bids, 1):
                records.append({
                    "experiment_id": experiment_id,
                    "round": round_num,
                    "player": player,
                    "team_id": team_id,
                    "bid": bid
                })
        
        if not records:
            logging.warning(f"No bidding records generated for {file_path}. Skipping.")
            return None

        df = pd.DataFrame(records)
        self.max_rounds_observed = max(self.max_rounds_observed, max_rounds_exp)

        # --- Calculate Targets and Team Averages ---
        targets = []
        team1_avgs = []
        team2_avgs = []
        for r in range(1, max_rounds_exp + 1):
            round_df = df[df['round'] == r]
            if round_df.empty:
                targets.append(np.nan)
                team1_avgs.append(np.nan)
                team2_avgs.append(np.nan)
                continue

            overall_avg = round_df['bid'].mean()
            targets.append(overall_avg * 0.8 if not np.isnan(overall_avg) else np.nan)

            t1_bids = round_df[round_df['team_id'] == 1]['bid']
            team1_avgs.append(t1_bids.mean() if not t1_bids.empty else np.nan)

            t2_bids = round_df[round_df['team_id'] == 2]['bid']
            team2_avgs.append(t2_bids.mean() if not t2_bids.empty else np.nan)

        round_summary = pd.DataFrame({
            'round': range(1, max_rounds_exp + 1),
            'target': targets,
            'team1_avg_bid': team1_avgs,
            'team2_avg_bid': team2_avgs
        }).set_index('round')

        df = df.merge(round_summary, on='round', how='left')

        # --- Add Winner Info ---
        winners_dict = data.get('winners', {})
        def get_winner_team(round_num):
            round_str = str(round_num)
            win_list = winners_dict.get(round_str, [])
            if not win_list: return 0 # Tie or no data
            if set(win_list).issubset(set(team1)): return 1
            if set(win_list).issubset(set(team2)): return 2
            logging.warning(f"Round {round_num} winners {win_list} in {experiment_id} don't match teams. Treating as Tie.")
            return 0 # Treat ambiguous cases as Tie

        df['winning_team'] = df['round'].apply(get_winner_team)

        # Store team members for summary
        self.summary_stats["team1_members"].update(team1)
        self.summary_stats["team2_members"].update(team2)

        logging.info(f"Successfully processed {experiment_id} ({max_rounds_exp} rounds)")
        return df

    def parse_teams_from_filename(self, filename):
        """Parses team names from filename."""
        base = os.path.splitext(os.path.basename(filename))[0]
        try:
            team1_part, team2_part_with_exp = base.split('_VS_')
            team2_part = '_'.join(team2_part_with_exp.split('_')[:-1]) # Remove _expX
            team1 = [name.replace('_', ' ') for name in team1_part.split('_')]
            team2 = [name.replace('_', ' ') for name in team2_part.split('_')]
            return team1, team2
        except ValueError:
            logging.error(f"Could not parse teams from filename '{filename}'. Using defaults.")
            return ["Unknown_Team1"], ["Unknown_Team2"]

    def aggregate_results(self):
        """Combines data from all experiments into a single DataFrame."""
        if not self.all_experiment_data:
            logging.error("No experiment data loaded to aggregate.")
            return None
        
        full_df = pd.concat(self.all_experiment_data, ignore_index=True)
        logging.info(f"Aggregated data from {len(self.all_experiment_data)} experiments.")
        return full_df

    def calculate_summary_stats(self, full_df):
        """Calculates aggregate statistics from the combined DataFrame."""
        if full_df is None or full_df.empty:
            logging.error("Cannot calculate summary stats on empty data.")
            return

        self.summary_stats["total_experiments"] = full_df['experiment_id'].nunique()
        
        # Calculate wins per team per experiment, then sum
        wins_per_exp = full_df.drop_duplicates(subset=['experiment_id', 'round'])[['experiment_id', 'round', 'winning_team']]
        win_counts = wins_per_exp['winning_team'].value_counts()

        self.summary_stats["team1_wins"] = win_counts.get(1, 0)
        self.summary_stats["team2_wins"] = win_counts.get(2, 0)
        self.summary_stats["ties"] = win_counts.get(0, 0)
        self.summary_stats["total_rounds_processed"] = len(wins_per_exp)

        # Calculate average deviations
        full_df['deviation'] = (full_df['bid'] - full_df['target']).abs()
        team1_dev = full_df[full_df['team_id'] == 1]['deviation'].mean()
        team2_dev = full_df[full_df['team_id'] == 2]['deviation'].mean()
        self.summary_stats["team1_avg_deviation_player"] = team1_dev if not np.isnan(team1_dev) else 0
        self.summary_stats["team2_avg_deviation_player"] = team2_dev if not np.isnan(team2_dev) else 0
        
        # Calculate average TEAM deviation from target
        # Ensure experiment_id is kept for later merging
        team_avg_dev = full_df.drop_duplicates(subset=['experiment_id', 'round'])[['experiment_id', 'round', 'team1_avg_bid', 'team2_avg_bid', 'target']].copy()
        team_avg_dev['team1_dev'] = (team_avg_dev['team1_avg_bid'] - team_avg_dev['target']).abs()
        team_avg_dev['team2_dev'] = (team_avg_dev['team2_avg_bid'] - team_avg_dev['target']).abs()
        self.summary_stats["team1_avg_deviation_team"] = team_avg_dev['team1_dev'].mean() if not team_avg_dev['team1_dev'].isnull().all() else np.nan
        self.summary_stats["team2_avg_deviation_team"] = team_avg_dev['team2_dev'].mean() if not team_avg_dev['team2_dev'].isnull().all() else np.nan

        # --- Calculate Persona Performance ---
        persona_dev = full_df.groupby('player')['deviation'].agg(['mean', 'count'])
        self.persona_performance_summary = persona_dev.rename(columns={'mean': 'avg_deviation', 'count': 'rounds_played'}).sort_values('avg_deviation')

        # --- Calculate Correlations ---
        # Prepare data for correlation: round-level averages
        round_agg = full_df.drop_duplicates(subset=['experiment_id', 'round']).copy()
        # Calculate intra-team std dev per round/exp/team
        intra_team_std = full_df.groupby(['experiment_id', 'round', 'team_id'])['bid'].std().reset_index().rename(columns={'bid': 'intra_team_std'})
        # Merge std dev back
        round_agg = round_agg.merge(intra_team_std[intra_team_std['team_id']==1][['experiment_id', 'round', 'intra_team_std']].rename(columns={'intra_team_std': 'team1_intra_std'}), on=['experiment_id', 'round'], how='left')
        round_agg = round_agg.merge(intra_team_std[intra_team_std['team_id']==2][['experiment_id', 'round', 'intra_team_std']].rename(columns={'intra_team_std': 'team2_intra_std'}), on=['experiment_id', 'round'], how='left')
        # Merge team deviation
        round_agg = round_agg.merge(team_avg_dev[['experiment_id', 'round', 'team1_dev', 'team2_dev']], on=['experiment_id', 'round'], how='left') # Need exp_id in team_avg_dev first
        # Add win outcome (binary for correlation)
        round_agg['team1_won'] = (round_agg['winning_team'] == 1).astype(int)
        round_agg['team2_won'] = (round_agg['winning_team'] == 2).astype(int)

        # Select columns for correlation
        corr_cols_t1 = ['team1_intra_std', 'team1_dev', 'team1_won']
        corr_cols_t2 = ['team2_intra_std', 'team2_dev', 'team2_won']
        
        # Calculate correlations, handling potential NaN columns
        self.correlation_data['team1'] = round_agg[corr_cols_t1].corr(method='pearson')
        self.correlation_data['team2'] = round_agg[corr_cols_t2].corr(method='pearson')


        # --- Chi-squared test for win difference significance ---
        t1_wins = self.summary_stats["team1_wins"]
        t2_wins = self.summary_stats["team2_wins"]
        observed = [[t1_wins, t2_wins]] # Needs to be 2D for chi2_contingency

        # Check if we have enough data for a meaningful test
        if t1_wins + t2_wins >= 5 and t1_wins >= 0 and t2_wins >= 0: # Basic check for validity
            try:
                chi2, p, _, expected = chi2_contingency(observed)
                # Check expected frequencies (rule of thumb: all should be >= 5)
                if np.all(expected >= 5):
                    self.summary_stats["chi2_statistic"] = chi2
                    self.summary_stats["p_value"] = p
                    logging.info(f"Chi-squared test performed: chi2={chi2:.4f}, p={p:.4f}")
                else:
                    logging.warning(f"Chi-squared test expected frequencies low ({expected.flatten()}). Results may be inaccurate.")
                    self.summary_stats["chi2_statistic"] = chi2 # Report anyway but warn
                    self.summary_stats["p_value"] = p
            except ValueError as e:
                 logging.error(f"Chi-squared test failed: {e}. Observed data: {observed}")
                 # Leave as NaN (default)
        else:
            logging.warning(f"Skipping Chi-squared test due to insufficient data (T1 Wins: {t1_wins}, T2 Wins: {t2_wins}).")
            # Leave as NaN (default)

        # --- Calculate Bid Distribution Stats (Mean, Std Dev, Normality) ---
        team1_bids = full_df[full_df['team_id'] == 1]['bid'].dropna()
        team2_bids = full_df[full_df['team_id'] == 2]['bid'].dropna()

        if not team1_bids.empty:
            self.summary_stats["team1_bid_mean"] = team1_bids.mean()
            self.summary_stats["team1_bid_std"] = team1_bids.std()
            # Shapiro-Wilk test requires at least 3 samples
            if len(team1_bids) >= 3:
                shapiro_test = shapiro(team1_bids)
                self.summary_stats["team1_shapiro_p"] = shapiro_test.pvalue
            else:
                self.summary_stats["team1_shapiro_p"] = np.nan
                logging.warning("Team 1 has less than 3 bids, cannot perform Shapiro-Wilk test.")
        else:
            self.summary_stats["team1_bid_mean"] = np.nan
            self.summary_stats["team1_bid_std"] = np.nan
            self.summary_stats["team1_shapiro_p"] = np.nan

        if not team2_bids.empty:
            self.summary_stats["team2_bid_mean"] = team2_bids.mean()
            self.summary_stats["team2_bid_std"] = team2_bids.std()
            if len(team2_bids) >= 3:
                shapiro_test = shapiro(team2_bids)
                self.summary_stats["team2_shapiro_p"] = shapiro_test.pvalue
            else:
                self.summary_stats["team2_shapiro_p"] = np.nan
                logging.warning("Team 2 has less than 3 bids, cannot perform Shapiro-Wilk test.")
        else:
            self.summary_stats["team2_bid_mean"] = np.nan
            self.summary_stats["team2_bid_std"] = np.nan
            self.summary_stats["team2_shapiro_p"] = np.nan


    def generate_plots(self, full_df):
        import matplotlib as mpl
        mpl.rcParams.update({
                "xtick.labelsize": 30, # Increased from 14
                "ytick.labelsize": 30  # Increased from 14
            })
        """Generates all plots from the aggregated data."""
        if full_df is None or full_df.empty:
            logging.error("Cannot generate plots on empty data.")
            return

        rounds = range(1, self.max_rounds_observed + 1)
        team1_name = "Team 1" # Simplified names for plots
        team2_name = "Team 2"

        # --- Plot 1: Team Average vs Target (Aggregated) ---
        plt.figure(figsize=(12, 7))
        agg_team_avg = full_df.groupby('round')[['team1_avg_bid', 'team2_avg_bid', 'target']].mean().reset_index()
        plt.plot(agg_team_avg['round'], agg_team_avg['team1_avg_bid'], label=f'{team1_name} Avg. Bid', marker='o', linestyle='-')
        plt.plot(agg_team_avg['round'], agg_team_avg['team2_avg_bid'], label=f'{team2_name} Avg. Bid', marker='s', linestyle='-')
        plt.plot(agg_team_avg['round'], agg_team_avg['target'], label='Target (0.8 * Overall Avg)', marker='^', linestyle='--', color='red')
        plt.title('Average Team Bids vs. Target per Round (Aggregated Across Experiments)', fontsize=16)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Average Bid / Target Value', fontsize=14)
        plt.xticks(rounds)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'Aggregated_TeamAverage_vs_Target.pdf'), format = 'pdf')
        plt.close()

        # --- Plot 2: Winning Team per Round (Aggregated Frequency) ---
        plt.figure(figsize=(12, 7))
        win_freq = full_df.drop_duplicates(subset=['experiment_id', 'round']) \
                          .groupby('round')['winning_team'].value_counts(normalize=True).unstack(fill_value=0)
        win_freq = win_freq.reindex(columns=[1, 2, 0], fill_value=0) # Ensure columns exist and order
        win_freq.plot(kind='bar', stacked=True, figsize=(12, 7),
                      color=[sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[2]],
                      edgecolor='black')
        plt.title('Frequency of Winning Team per Round (Aggregated)', fontsize=16)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Proportion of Experiments', fontsize=14)
        plt.xticks(rotation=0)
        plt.legend(title='Outcome', labels=[team1_name, team2_name, 'Tie'], loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend
        plt.savefig(os.path.join(self.output_dir, 'Aggregated_Winning_Team_Frequency.pdf'), format = "pdf")
        plt.close()

        # --- Plot 3: Intra-Team Bid Convergence (Std Dev) ---
        # Calculate average std dev per round per team
        intra_team_std = full_df.groupby(
            ['experiment_id', 'round', 'team_id']
        )['bid'].std(ddof=0).reset_index()
        avg_intra_team_std = intra_team_std.groupby(
            ['round', 'team_id']
        )['bid'].mean().unstack()

        if not avg_intra_team_std.empty:
            plt.figure(figsize=(12, 7))
            # turn off all grid lines
            plt.grid(False)

            # choose a thicker line width
            lw = 3

            # plot team 1
            if 1 in avg_intra_team_std.columns:
                plt.plot(
                    avg_intra_team_std.index,
                    avg_intra_team_std[1].fillna(0),
                    label=f'{team1_name} Avg. Intra-Team Std Dev',
                    marker='o',
                    linewidth=lw,
                    markersize=8
                )
            # plot team 2
            if 2 in avg_intra_team_std.columns:
                plt.plot(
                    avg_intra_team_std.index,
                    avg_intra_team_std[2].fillna(0),
                    label=f'{team2_name} Avg. Intra-Team Std Dev',
                    marker='s',
                    linewidth=lw,
                    markersize=8
                )
            plt.xticks(rounds)
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, 'Aggregated_IntraTeam_Convergence.pdf'),
                format = "pdf"
            )
            plt.close()
        else:
            logging.warning("Could not generate Intra-Team Convergence plot due to missing data.")


        # --- Plot 4: Bid Distribution per Team ---
        bid_data = full_df[full_df['team_id'] != 0] # Exclude potential non-team players if any
        if not bid_data.empty:
            plt.figure(figsize=(12, 7))
            sns.histplot(data=bid_data, x='bid', hue='team_id', kde=True, palette='viridis', element="step", stat="density", common_norm=False)
            plt.title('Distribution of Individual Bids per Team (All Rounds)', fontsize=16)
            plt.xlabel('Bid Value', fontsize=14)
            plt.ylabel('Density', fontsize=14) # Changed from Frequency
            # Manually set legend labels if needed, seaborn might handle it
            handles, labels = plt.gca().get_legend_handles_labels()
            if len(labels) >= 2: # Check if legend items exist
                 # Map team_id (e.g., 1.0, 2.0) back to names if needed, or use defaults
                 legend_map = {'1.0': team1_name, '2.0': team2_name, '1': team1_name, '2': team2_name}
                 new_labels = [legend_map.get(lbl, lbl) for lbl in labels]
                 plt.legend(title='Team', handles=handles, labels=new_labels)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'Aggregated_Bid_Distribution.pdf'), format = "pdf")
            plt.close()
        else:
             logging.warning("Could not generate Bid Distribution plot due to missing data.")


        # --- Plot 5: Average Deviation from Target per Round ---
        # Calculate average team deviation per round
        team_avg_dev_per_round = full_df.drop_duplicates(subset=['experiment_id', 'round'])[['experiment_id', 'round', 'team1_avg_bid', 'team2_avg_bid', 'target']].copy()
        team_avg_dev_per_round['team1_dev'] = (team_avg_dev_per_round['team1_avg_bid'] - team_avg_dev_per_round['target']).abs()
        team_avg_dev_per_round['team2_dev'] = (team_avg_dev_per_round['team2_avg_bid'] - team_avg_dev_per_round['target']).abs()
        avg_dev_plot_data = team_avg_dev_per_round.groupby('round')[['team1_dev', 'team2_dev']].mean().reset_index()

        if not avg_dev_plot_data.empty:
            plt.figure(figsize=(12, 7))
            plt.plot(avg_dev_plot_data['round'], avg_dev_plot_data['team1_dev'], label=f'{team1_name} Avg. Deviation', marker='o')
            plt.plot(avg_dev_plot_data['round'], avg_dev_plot_data['team2_dev'], label=f'{team2_name} Avg. Deviation', marker='s')
            plt.title('Average Team Deviation from Target per Round', fontsize=16)
            plt.xlabel('Round', fontsize=14)
            plt.ylabel('Average Absolute Deviation', fontsize=14)
            plt.xticks(rounds)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'Aggregated_Team_Deviation_Per_Round.pdf'), format = "pdf")
            plt.close()
        else:
            logging.warning("Could not generate Team Deviation per Round plot due to missing data.")

        # --- Plot 6: Persona Average Deviation ---
        if not self.persona_performance_summary.empty:
             # Limit to top N personas if too many
            max_personas_to_plot = 20
            plot_data = self.persona_performance_summary.head(max_personas_to_plot)

            plt.figure(figsize=(max(10, len(plot_data) * 0.5), 7)) # Adjust width
            sns.barplot(x=plot_data.index, y=plot_data['avg_deviation'], palette='viridis')
            plt.title(f'Average Deviation from Target per Persona (Top {len(plot_data)})', fontsize=16)
            plt.xlabel('Persona', fontsize=14)
            plt.ylabel('Average Absolute Deviation from Target', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate labels
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'Persona_Avg_Deviation.pdf'), format = "pdf")
            plt.close()
        else:
             logging.warning("Could not generate Persona Average Deviation plot due to missing data.")

        # --- Plot 7: Gaussian Fit Plots per Team ---
        team1_bids = full_df[full_df['team_id'] == 1]['bid'].dropna()
        team2_bids = full_df[full_df['team_id'] == 2]['bid'].dropna()

        for team_id, bids, team_label in [(1, team1_bids, team1_name), (2, team2_bids, team2_name)]:
            if not bids.empty and len(bids) > 1: # Need at least 2 points for std dev
                mean = self.summary_stats.get(f"team{team_id}_bid_mean", np.nan)
                std = self.summary_stats.get(f"team{team_id}_bid_std", np.nan)

                if not np.isnan(mean) and not np.isnan(std) and std > 0:
                    plt.figure(figsize=(10, 6))
                    # Plot histogram
                    sns.histplot(bids, kde=False, stat="density", bins=15, label='Bid Histogram', color=sns.color_palette()[team_id-1], alpha=0.6)

                    # Plot Gaussian fit
                    xmin, xmax = plt.xlim()
                    x = np.linspace(xmin, xmax, 100)
                    p = norm.pdf(x, mean, std)
                    plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit (μ={mean:.2f}, σ={std:.2f})')

                    plt.title(f'{team_label} Bid Distribution with Gaussian Fit', fontsize=16)
                    plt.xlabel('Bid Value', fontsize=14)
                    plt.ylabel('Density', fontsize=14)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f'Team{team_id}_Bid_Gaussian_Fit.pdf'), format = "pdf")
                    plt.close()
                else:
                    logging.warning(f"Could not generate Gaussian fit plot for {team_label} due to invalid mean/std (Mean: {mean}, Std: {std}).")
            else:
                logging.warning(f"Could not generate Gaussian fit plot for {team_label} due to insufficient data (count: {len(bids)}).")


        logging.info(f"All aggregated plots saved to '{self.output_dir}'.")


    def generate_summary_report(self):
        """Generates and saves textual and CSV summaries."""
        if not self.summary_stats or self.summary_stats.get("total_experiments", 0) == 0:
            logging.warning("No summary stats calculated or no experiments processed. Skipping report generation.")
            return

        stats = self.summary_stats # Alias for brevity
        total_outcomes = stats.get("team1_wins", 0) + stats.get("team2_wins", 0) + stats.get("ties", 0)
        team1_win_rate = (stats.get("team1_wins", 0) / total_outcomes * 100) if total_outcomes > 0 else 0
        team2_win_rate = (stats.get("team2_wins", 0) / total_outcomes * 100) if total_outcomes > 0 else 0
        tie_rate = (stats.get("ties", 0) / total_outcomes * 100) if total_outcomes > 0 else 0

        team1_names = ", ".join(sorted(list(stats.get("team1_members", set())))) or "Team 1"
        team2_names = ", ".join(sorted(list(stats.get("team2_members", set())))) or "Team 2"

        # Chi-squared interpretation
        p_value = stats.get("p_value", np.nan)
        chi2_stat = stats.get("chi2_statistic", np.nan)
        if np.isnan(p_value):
            sig_test_result = "N/A (Test not performed or failed)"
        elif p_value < 0.05:
            sig_test_result = f"Statistically Significant (p={p_value:.4f}, chi2={chi2_stat:.4f})"
        else:
            sig_test_result = f"Not Statistically Significant (p={p_value:.4f}, chi2={chi2_stat:.4f})"

        # --- Text Summary ---
        summary_text = f"""
Evaluation Summary Report
=========================
Experiments Evaluated: {stats.get("total_experiments", 0)}
Total Rounds Processed: {stats.get("total_rounds_processed", 0)}
Max Rounds Observed:    {self.max_rounds_observed}

Overall Performance:
--------------------
Team 1 ({team1_names}):
  - Wins:                 {stats.get("team1_wins", 0)} ({team1_win_rate:.2f}%)
  - Wins:                 {stats.get("team1_wins", 0)} ({team1_win_rate:.2f}%)
  - Avg. Team Deviation:  {stats.get("team1_avg_deviation_team", np.nan):.4f} (Team Avg Bid vs Target)
  - Avg. Player Deviation:{stats.get("team1_avg_deviation_player", np.nan):.4f} (Individual Bid vs Target)
  - Bid Mean:             {stats.get("team1_bid_mean", np.nan):.4f}
  - Bid Std Dev:          {stats.get("team1_bid_std", np.nan):.4f}
  - Shapiro-Wilk p-value: {stats.get("team1_shapiro_p", np.nan):.4f} (Normality Test)

Team 2 ({team2_names}):
  - Wins:                 {stats.get("team2_wins", 0)} ({team2_win_rate:.2f}%)
  - Avg. Team Deviation:  {stats.get("team2_avg_deviation_team", np.nan):.4f} (Team Avg Bid vs Target)
  - Avg. Player Deviation:{stats.get("team2_avg_deviation_player", np.nan):.4f} (Individual Bid vs Target)
  - Bid Mean:             {stats.get("team2_bid_mean", np.nan):.4f}
  - Bid Std Dev:          {stats.get("team2_bid_std", np.nan):.4f}
  - Shapiro-Wilk p-value: {stats.get("team2_shapiro_p", np.nan):.4f} (Normality Test)

Ties: {stats.get("ties", 0)} ({tie_rate:.2f}%)

Statistical Significance of Win Difference (Team 1 vs Team 2):
  - Chi-squared Test: {sig_test_result}

Persona Performance (Avg. Deviation from Target):
-------------------------------------------------
"""
        # Add top/bottom personas
        if not self.persona_performance_summary.empty:
            top_n = 5
            summary_text += f"Top {top_n} Personas (Lowest Deviation):\n"
            for name, data in self.persona_performance_summary.head(top_n).iterrows():
                summary_text += f"  - {name}: {data['avg_deviation']:.4f} (over {int(data['rounds_played'])} rounds)\n"
            summary_text += f"\nBottom {top_n} Personas (Highest Deviation):\n"
            for name, data in self.persona_performance_summary.tail(top_n).iloc[::-1].iterrows(): # Reverse tail
                summary_text += f"  - {name}: {data['avg_deviation']:.4f} (over {int(data['rounds_played'])} rounds)\n"
        else:
            summary_text += "  N/A (Could not calculate persona performance)\n"

        summary_text += f"""
Correlation Analysis (Pearson's r):
-----------------------------------
Team 1 Correlations:
{self.correlation_data.get('team1', pd.DataFrame()).to_string(float_format="%.3f")}

Team 2 Correlations:
{self.correlation_data.get('team2', pd.DataFrame()).to_string(float_format="%.3f")}

(Note: Correlations calculated on round-level aggregated data across experiments)
=========================
"""
        print(summary_text)
        summary_file_path = os.path.join(self.output_dir, "evaluation_summary.txt")
        try:
            with open(summary_file_path, "w") as f:
                f.write(summary_text)
            logging.info(f"Text summary report saved to: {summary_file_path}")
        except IOError as e:
            logging.error(f"Failed to save text summary report: {e}")

        # --- CSV Summary ---
        # Basic stats
        csv_metrics = {
            "Total Experiments": stats.get("total_experiments", 0),
            "Total Rounds Processed": stats.get("total_rounds_processed", 0),
            "Max Rounds Observed": self.max_rounds_observed,
            "Team 1 Members": team1_names,
            "Team 1 Wins": stats.get("team1_wins", 0),
            "Team 1 Win Rate (%)": f"{team1_win_rate:.2f}",
            "Team 1 Avg Team Deviation": f"{stats.get('team1_avg_deviation_team', np.nan):.4f}",
            "Team 1 Avg Player Deviation": f"{stats.get('team1_avg_deviation_player', np.nan):.4f}",
            "Team 1 Bid Mean": f"{stats.get('team1_bid_mean', np.nan):.4f}",
            "Team 1 Bid Std Dev": f"{stats.get('team1_bid_std', np.nan):.4f}",
            "Team 1 Shapiro-Wilk p-value": f"{stats.get('team1_shapiro_p', np.nan):.4f}",
            "Team 2 Members": team2_names,
            "Team 2 Wins": stats.get("team2_wins", 0),
            "Team 2 Win Rate (%)": f"{team2_win_rate:.2f}",
            "Team 2 Avg Team Deviation": f"{stats.get('team2_avg_deviation_team', np.nan):.4f}",
            "Team 2 Avg Player Deviation": f"{stats.get('team2_avg_deviation_player', np.nan):.4f}",
            "Team 2 Bid Mean": f"{stats.get('team2_bid_mean', np.nan):.4f}",
            "Team 2 Bid Std Dev": f"{stats.get('team2_bid_std', np.nan):.4f}",
            "Team 2 Shapiro-Wilk p-value": f"{stats.get('team2_shapiro_p', np.nan):.4f}",
            "Ties": stats.get("ties", 0),
            "Tie Rate (%)": f"{tie_rate:.2f}",
            "Chi2 Statistic (Win Diff)": f"{chi2_stat:.4f}" if not np.isnan(chi2_stat) else "N/A",
            "P-value (Win Diff)": f"{p_value:.4f}" if not np.isnan(p_value) else "N/A",
            "Win Difference Significant (p<0.05)": "Yes" if not np.isnan(p_value) and p_value < 0.05 else ("No" if not np.isnan(p_value) else "N/A")
        }
        # Add correlations
        for team_label, corr_matrix in self.correlation_data.items():
            if not corr_matrix.empty:
                for col in corr_matrix.columns:
                    for idx in corr_matrix.index:
                        if idx != col: # Avoid self-correlation
                             metric_name = f"Corr ({team_label}): {idx} vs {col}"
                             csv_metrics[metric_name] = f"{corr_matrix.loc[idx, col]:.3f}" if not pd.isna(corr_matrix.loc[idx, col]) else "N/A"

        # Add persona performance
        if not self.persona_performance_summary.empty:
             for name, data in self.persona_performance_summary.iterrows():
                  csv_metrics[f"Persona Avg Deviation: {name}"] = f"{data['avg_deviation']:.4f}"
                  csv_metrics[f"Persona Rounds Played: {name}"] = int(data['rounds_played'])


        # Convert dict to DataFrame for saving
        summary_df = pd.DataFrame(list(csv_metrics.items()), columns=['Metric', 'Value'])
        csv_file_path = os.path.join(self.output_dir, "evaluation_summary.csv")
        try:
            summary_df.to_csv(csv_file_path, index=False)
            logging.info(f"CSV summary report saved to: {csv_file_path}")
        except IOError as e:
            logging.error(f"Failed to save CSV summary report: {e}")


    def run_evaluation(self):
        """Main method to run the full evaluation process."""
        experiment_files = sorted(glob(os.path.join(self.result_dir, '*.json')))
        if not experiment_files:
            logging.warning(f"No JSON files found in '{self.result_dir}'.")
            return

        files_to_process = experiment_files[:self.num_experiments]
        logging.info(f"Found {len(experiment_files)} experiments, processing {len(files_to_process)}.")

        for exp_file in files_to_process:
            processed_data = self.load_and_process_experiment(exp_file)
            if processed_data is not None:
                self.all_experiment_data.append(processed_data)

        if not self.all_experiment_data:
             logging.error("No experiments were successfully processed. Aborting further analysis.")
             return

        # Aggregate data after processing all files
        full_df = self.aggregate_results()

        if full_df is not None:
            # Calculate overall summary statistics
            self.calculate_summary_stats(full_df)

            # Generate plots based on aggregated data
            self.generate_plots(full_df)

            # Generate text and CSV summary reports
            self.generate_summary_report()
        else:
             logging.error("Aggregation failed. Cannot proceed with plotting and summary.")


def main():
    parser = argparse.ArgumentParser(description='Evaluate G08A game experiments for conference paper analysis.')
    parser.add_argument('--result_dir', type=str, default='result',
                        help='Directory containing experiment JSON result files.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results_paper',
                        help='Directory to save evaluation plots and summaries.')
    parser.add_argument('--num_experiments', type=int, default=50, # Default to a larger number for paper analysis
                        help='Maximum number of experiments to evaluate.')

    args = parser.parse_args()

    # --- Check for dependencies ---
    try:
        import pandas
        import seaborn
        import scipy
    except ImportError as e:
        logging.error(f"Missing required dependency: {e.name}. Please install pandas, seaborn, and scipy.")
        logging.error("You can usually install them using: pip install pandas seaborn scipy")
        return # Exit if dependencies are missing

    evaluator = G08AEvaluator(
        result_dir=args.result_dir,
        output_dir=args.output_dir,
        num_experiments=args.num_experiments
    )
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
