import pandas as pd
import matplotlib.pyplot as plt

baseline_csv = 'metrics_baseline.csv'
dividemix_csv = 'training_history.csv'
baseline_data = pd.read_csv(baseline_csv)
dividemix_data = pd.read_csv(dividemix_csv)

baseline_data['model'] = 'Baseline'
dividemix_data['model'] = 'DivideMix'

data = pd.concat([baseline_data, dividemix_data])

colors = {
    'Baseline Train Loss': '#1f77b4',  # Blue
    'Baseline Test Loss': '#aec7e8',   # Light Blue
    'DivideMix Train Loss': '#ff7f0e', # Orange
    'DivideMix Test Loss': '#ffbb78',  # Light Orange
    'Baseline Train Accuracy': '#2ca02c', # Green
    'Baseline Test Accuracy': '#98df8a',  # Light Green
    'DivideMix Train Accuracy': '#d62728', # Red
    'DivideMix Test Accuracy': '#ff9896'   # Light Red
}

plt.figure()
for model in data['model'].unique():
    subset = data[data['model'] == model]
    plt.plot(subset['epoch'], subset['train_loss'], label=f'{model} Train Loss', color=colors[f'{model} Train Loss'])
    plt.plot(subset['epoch'], subset['test_loss'], label=f'{model} Test Loss', color=colors[f'{model} Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid()
plt.savefig('loss_comparison.png')
plt.show()

plt.figure()
for model in data['model'].unique():
    subset = data[data['model'] == model]
    plt.plot(subset['epoch'], subset['train_acc'], label=f'{model} Train Accuracy', color=colors[f'{model} Train Accuracy'])
    plt.plot(subset['epoch'], subset['test_acc'], label=f'{model} Test Accuracy', color=colors[f'{model} Test Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy Over Epochs')
plt.legend()
plt.grid()
plt.savefig('accuracy_comparison.png')
plt.show()


baseline_stats = {
    'Total Training Time (s)': 10086.03,
    'Peak GPU Memory (GB)': 2.76,
    'Inference Time (s)': 16.51
}
dividemix_stats = {
    'Total Training Time (s)': 99965.90,
    'Peak GPU Memory (GB)': 0.917,
    'Inference Time (s)': 28.76
}

summary_data = {
    'Metric': list(baseline_stats.keys()),
    'Baseline': list(baseline_stats.values()),
    'DivideMix': list(dividemix_stats.values())
}
summary_df = pd.DataFrame(summary_data)

for metric in summary_df['Metric']:
    plt.figure()
    plt.bar(['Baseline', 'DivideMix'], summary_df[summary_df['Metric'] == metric].iloc[0, 1:])
    plt.ylabel(metric)
    plt.title(f'Comparison of {metric}')
    plt.savefig(f'{metric.replace(" ", "_").lower()}_comparison.png')
    plt.show()

print("Summary Metrics")
print(summary_df.to_markdown(index=False))