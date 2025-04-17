import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy.stats import f_oneway

issue_analysis_path = r'datasets\issue_topic_analysis\issue_analysis_results.json'
with open(issue_analysis_path, 'r') as file:
    issue_analysis = json.load(file)

issue_analysis_flat = []
for item in issue_analysis:
    for topic in item['top_topics']:
        flat_item = item.copy()
        flat_item['top_topics'] = topic['topic_id'] 
        issue_analysis_flat.append(flat_item)

df = pd.DataFrame(issue_analysis_flat)

df['issue_type'] = df['issue_type'].apply(lambda x: x['type'] if isinstance(x, dict) else x)

def calculate_statistics(data):
    stats = {
        'Mean': data.mean(),
        'Median': data.median(),
        'Std Deviation': data.std()
    }
    return pd.Series(stats)

grouped = df.groupby(['top_topics', 'issue_type'])


fields = ['n_comments', 'n_attachment', 'total_comment_length']
topic_stats = {}
for field in fields:
    topic_stats[field] = grouped[field].apply(calculate_statistics)

anova_results = {}
for field in fields:
    anova_results[field] = {}
    f_statistic, p_value = f_oneway(*[group[field] for name, group in grouped])
    anova_results[field]['F-statistic'] = f_statistic
    anova_results[field]['P-value'] = p_value

plots_folder = os.path.join('datasets', 'RQ3', 'plots')
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

for field, data in topic_stats.items():
    print(f"Statistics for {field}:")
    print(data)
    print()
    q1 = df[field].quantile(0.25)
    q3 = df[field].quantile(0.75)
    iqr = q3 - q1
    whisker_range = 0.7 
    lower_whisker = q1 - whisker_range * iqr
    upper_whisker = q3 + whisker_range * iqr
    filtered_data = df[(df[field] >= lower_whisker) & (df[field] <= upper_whisker)]

    plt.figure(figsize=(14, 8))
    sns.boxplot(x='top_topics', y=field, hue='issue_type', data=filtered_data)
    plt.title(f'Distribution of {field} by Topic and Issue Type')
    plt.xlabel('Topic')
    plt.ylabel(field)
    plt.legend(title='Issue Type')
    plt.xticks(rotation=45) 
    plt.tight_layout()  
    plot_filename = os.path.join(plots_folder, f'{field}_boxplot.png')
    plt.savefig(plot_filename)
    plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='n_comments', y='total_comment_length', hue='issue_type')
plt.title('Scatter Plot of Number of Comments vs. Total Comment Length')
plt.xlabel('Number of Comments')
plt.ylabel('Total Comment Length')
plt.legend(title='Issue Type')
plot_filename = os.path.join(plots_folder, 'scatterplot_comments_vs_length.png')
plt.savefig(plot_filename)
plt.close()

# Histograms
plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
sns.histplot(df['n_comments'], kde=True)
plt.title('Histogram of Number of Comments')

plt.subplot(1, 3, 2)
sns.histplot(df['n_attachment'], kde=True)
plt.title('Histogram of Number of Attachments')

plt.subplot(1, 3, 3)
sns.histplot(df['total_comment_length'], kde=True)
plt.title('Histogram of Total Comment Length')

plt.tight_layout()
plot_filename = os.path.join(plots_folder, 'histograms.png')
plt.savefig(plot_filename)
plt.close()

plt.figure(figsize=(10, 6))
sns.heatmap(df[['n_comments', 'n_attachment', 'total_comment_length']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Variables')
plot_filename = os.path.join(plots_folder, 'correlation_heatmap.png')
plt.savefig(plot_filename)
plt.close()

with open(os.path.join(plots_folder, 'anova_results.json'), 'w') as json_file:
    json.dump(anova_results, json_file)

print("ANOVA Test Results:")
print(json.dumps(anova_results, indent=4))
