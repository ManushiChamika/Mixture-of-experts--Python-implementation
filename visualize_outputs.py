import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

# Set the style and figure size for better visibility
# Use seaborn to set theme (avoid matplotlib style lookup failures)
sns.set()
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

# Load the training log data
with open('moe_outputs/training_log.txt', 'r') as f:
    log_data = [json.loads(line) for line in f]

# Extract data from the log and coerce to numeric numpy arrays
epochs = np.array([entry['epoch'] for entry in log_data], dtype=float)
entropy = np.array([entry['entropy'] for entry in log_data], dtype=float)
routing_acc = np.array([entry['routing_acc'] for entry in log_data], dtype=float)
expert_utilization_over_time = np.array([entry['utilization'] for entry in log_data], dtype=float)
if expert_utilization_over_time.ndim == 1:
    # single epoch -> reshape to (1, num_experts)
    expert_utilization_over_time = expert_utilization_over_time.reshape(1, -1)
num_experts = expert_utilization_over_time.shape[1]  # Number of experts

# Plot expert utilization over time
plt.figure()
# Use integer x positions to avoid odd autoscaling issues; labels show actual epoch numbers
x = np.arange(len(epochs))
for i in range(num_experts):
    plt.plot(x, expert_utilization_over_time[:, i] * 100,
             marker='o', linewidth=2.5, markersize=6, label=f'Expert {i}')
plt.title('Expert Utilization Over Time', fontsize=12, pad=10)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Utilization %', fontsize=10)
plt.xticks(x, epochs.astype(int))
plt.xlim(x[0] - 0.5, x[-1] + 0.5)
plt.ylim(0, 100)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('moe_outputs/expert_utilization.png', bbox_inches='tight', dpi=150)
plt.close()

# Create router entropy plot
plt.figure()
plt.plot(x, entropy, marker='o', color='blue', linewidth=2.5)
plt.title('Router Entropy over Training', fontsize=12, pad=10)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Entropy', fontsize=10)
plt.xticks(x, epochs.astype(int))
plt.xlim(x[0] - 0.5, x[-1] + 0.5)
plt.ylim(max(0, entropy.min() - 0.1), entropy.max() + 0.1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('moe_outputs/router_entropy.png', dpi=150)
plt.close()

# Create routing accuracy plot
plt.figure()
plt.plot(x, routing_acc, marker='o', color='green', linewidth=2.5)
plt.title('Routing Accuracy over Training', fontsize=12, pad=10)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Routing Accuracy', fontsize=10)
plt.xticks(x, epochs.astype(int))
plt.xlim(x[0] - 0.5, x[-1] + 0.5)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('moe_outputs/routing_accuracy.png', dpi=150)
plt.close()

# Load and plot the router affinity matrix
file_path = 'moe_outputs/router_affinity_matrix.csv'
data = pd.read_csv(file_path, header=None)

# Check if data is loaded correctly
if data.empty:
    raise ValueError('The data is empty. Please check the CSV file.')

# Set the style of seaborn
sns.set(style='whitegrid')

# Create a heatmap for the router affinity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Router Affinity Matrix Heatmap')
plt.xlabel('Experts')
plt.ylabel('Samples')
plt.savefig('router_affinity_matrix_heatmap.png')
plt.close()

# Calculate aggregate expert utilization (sum of affinities for each expert)
expert_utilization = data.sum()
plt.figure(figsize=(10, 6))
expert_utilization.plot(kind='bar')
plt.title('Aggregate Expert Utilization')
plt.xlabel('Expert Index')
plt.ylabel('Total Utilization')
plt.tight_layout()
# Save to a different filename so it doesn't overwrite the time-series plot
plt.savefig('moe_outputs/expert_utilization_by_sum.png', dpi=150)
plt.close()