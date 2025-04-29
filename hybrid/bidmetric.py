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

# Seaborn theme and context for very large figure text
sns.set_theme(style="whitegrid")
sns.set_context("paper", rc={
    'font.size': 28,            # Increased base font size
    'axes.titlesize': 36,       # Increased title size
    'axes.labelsize': 32,       # Increased label size
    'xtick.labelsize': 30,      # Increased tick label size
    'ytick.labelsize': 30,      # Increased tick label size
    'legend.fontsize': 30,      # Increased legend font size
    'legend.title_fontsize': 32,# Increased legend title size
    'figure.dpi': 600           # Increased DPI to 600 for sharpness
})

class G08AEvaluator:
    def __init__(self, result_dir, output_dir, num_experiments) -> None:
        self.result_dir = result_dir
        self.output_dir = output_dir
        self.num_experiments = num_experiments
        self.all_experiment_data = []
        self.summary_stats = defaultdict(lambda: np.nan)
        self.summary_stats["team1_members"] = set()
        self.summary_stats["team2_members"] = set()
        self.summary_stats["total_experiments"] = 0
        self.summary_stats["total_rounds_processed"] = 0
        self.summary_stats["team1_wins"] = 0
        self.summary_stats["team2_wins"] = 0
        self.summary_stats["ties"] = 0
        self.correlation_data = {}
        self.max_rounds_observed = 0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(f"Created output directory: {self.output_dir}")

    def load_and_process_experiment(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load JSON from {file_path}: {e}")
            return None

        if not all(k in data for k in ['winners', 'biddings']):
            logging.error(f"Missing keys in {file_path}. Skipping.")
            return None
        if not data['biddings']:
            logging.warning(f"Empty biddings in {file_path}. Skipping.")
            return None

        experiment_id = os.path.splitext(os.path.basename(file_path))[0]
        team1, team2 = self.parse_teams_from_filename(file_path)
        records, max_rounds_exp = [], 0
        for player, bids in data['biddings'].items():
            max_rounds_exp = max(max_rounds_exp, len(bids))
            for i, bid in enumerate(bids, 1):
                name = player.replace(' Copy', ' Clone')
                records.append({
                    'experiment_id': experiment_id,
                    'round': i,
                    'player': name,
                    'bid': bid
                })
        if not records:
            logging.warning(f"No records for {file_path}. Skipping.")
            return None

        df = pd.DataFrame(records)
        self.max_rounds_observed = max(self.max_rounds_observed, max_rounds_exp)
        return df

    def parse_teams_from_filename(self, filename):
        base = os.path.splitext(os.path.basename(filename))[0]
        try:
            t1, t2exp = base.split('_VS_')
            t2 = '_'.join(t2exp.split('_')[:-1])
            return [x.replace('_', ' ') for x in t1.split('_')], [x.replace('_', ' ') for x in t2.split('_')]
        except:
            return [], []

    def aggregate_results(self):
        if not self.all_experiment_data:
            logging.error("No data to aggregate.")
            return None
        full = pd.concat(self.all_experiment_data, ignore_index=True)
        logging.info(f"Aggregated {len(self.all_experiment_data)} files.")
        return full

    def run_evaluation(self):
        files = sorted(glob(os.path.join(self.result_dir, '*.json')))[:self.num_experiments]
        for f in files:
            df = self.load_and_process_experiment(f)
            if df is not None:
                self.all_experiment_data.append(df)
        full_df = self.aggregate_results()
        if full_df is None:
            return

        # Compute agent-round average bids
        round_avg = full_df.groupby(['player', 'round'])['bid'].mean().reset_index()
        pivot = round_avg.pivot(index='round', columns='player', values='bid')
        pivot.to_csv(os.path.join(self.output_dir, 'agent_round_avg_bids.csv'), float_format='%.2f')
        self.agent_round_avg = pivot

        # Compute average per-run dispersion & range
        disp = full_df.groupby(['player','experiment_id'])['bid'].var(ddof=0).reset_index(name='sigma_bid')
        self.avg_dispersion = disp.groupby('player')['sigma_bid'].mean().sort_values()
        rng = full_df.groupby(['player','experiment_id'])['bid'].apply(lambda x: x.max()-x.min()).reset_index(name='range_bid')
        self.avg_range = rng.groupby('player')['range_bid'].mean().sort_values()

        self.generate_plots()

    def generate_plots(self):
        rounds = list(range(1, self.max_rounds_observed + 1))

        # Plot 1: Average Bid per Agent by Round
        if hasattr(self, 'agent_round_avg') and not self.agent_round_avg.empty:
            plt.figure(figsize=(16, 10))
            for agent in self.agent_round_avg.columns:
                plt.plot(
                    self.agent_round_avg.index,
                    self.agent_round_avg[agent],
                    marker='o', markersize=10,
                    linewidth=3, label=agent
                )
            plt.xticks(rounds)
            plt.legend(loc='best', frameon=False)
            plt.tight_layout()
            path_base = os.path.join(self.output_dir, 'Agent_Round_Average_Bids')
            plt.savefig(path_base + '.pdf', format='pdf')
            plt.savefig(path_base + '.png', dpi=600)
            plt.close()
        else:
            logging.warning("No agent-round averages to plot.")

        # Plot 2: Agent Bid Range (horizontal, sorted, with annotations)
        if hasattr(self, 'avg_range') and not self.avg_range.empty:
            plt.figure(figsize=(14, 10))
            sns.barplot(x=self.avg_range.values, y=self.avg_range.index, palette='mako')
            plt.xlabel('Average Range (max - min)', labelpad=15)
            plt.ylabel('')
            max_val = self.avg_range.max()
            for i, v in enumerate(self.avg_range.values):
                plt.text(v + max_val * 0.02, i, f"{v:.1f}", va='center', fontsize=26)
            sns.despine(left=False, bottom=False)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            path_base = os.path.join(self.output_dir, 'Agent_Bid_Range')
            plt.savefig(path_base + '.pdf', format='pdf')
            plt.savefig(path_base + '.png', dpi=600)
            plt.close()
        else:
            logging.warning("No agent-range data to plot.")

        # Plot 3: Agent Bid Dispersion (variance σ_bid)
        if hasattr(self, 'avg_dispersion') and not self.avg_dispersion.empty:
            plt.figure(figsize=(14, 10))
            sns.barplot(x=self.avg_dispersion.values, y=self.avg_dispersion.index, palette='rocket')
            plt.title('Average Bid Dispersion (σ_bid)', pad=20)
            plt.xlabel('Average Variance (σ_bid)', labelpad=15)
            plt.ylabel('')
            max_d = self.avg_dispersion.max()
            for i, v in enumerate(self.avg_dispersion.values):
                plt.text(v + max_d * 0.02, i, f"{v:.2f}", va='center', fontsize=26)
            sns.despine(left=False, bottom=False)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            path_base = os.path.join(self.output_dir, 'Agent_Bid_Dispersion')
            plt.savefig(path_base + '.pdf', format='pdf')
            plt.savefig(path_base + '.png', dpi=600)
            plt.close()
        else:
            logging.warning("No agent-dispersion data to plot.")

        logging.info("All plots saved in both PDF and PNG formats.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default='result')
    parser.add_argument('--output_dir', default='evaluation_results_paper')
    parser.add_argument('--num_experiments', type=int, default=50)
    args = parser.parse_args()
    G08AEvaluator(args.result_dir, args.output_dir, args.num_experiments).run_evaluation()
