import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
import json

# Load the Excel file
file_path = './datasets/Issues_2nd.xlsx'
df = pd.read_excel(file_path, sheet_name='HDFS')

# Load the topics from the JSON file
topics_data_path = r'datasets\lda_results\weightage_words\lda_result_iteration2_weightage_8.json'
with open(topics_data_path, 'r') as file:
    topics = json.load(file)

# Load the issues data from the JSON file
issues_data_path = 'datasets/pre_processed_data/cleaned_issues_data.json'
with open(issues_data_path, 'r') as file:
    issues_data = json.load(file)

# Extract top 6 words from topics
def extract_top_words(topic, n=5):
    words_with_weights = re.findall(r'(\d+\.\d+)\*\"([^\"]*)\"', topic["words"])
    sorted_words = sorted(words_with_weights, key=lambda x: float(x[0]), reverse=True)
    return [word for _, word in sorted_words[:n]]

topics_words = {topic["topic_id"]: extract_top_words(topic) for topic in topics}

# Function to check if an issue discusses a topic
def issue_discusses_topic(issue_text, topic_words):
    return any(word in issue_text for word in topic_words)

# Combine relevant fields for each issue and count issues per topic
manual_counts = Counter()
automatic_counts = Counter()

for _, row in df.iterrows():
    decision_type = row['Manual or automatic']
    issue_id = row['Issue key']
    issue = next((issue for issue in issues_data if issue['issue_id'] == issue_id), None)
    
    if issue:
        issue_text = issue['summary_description_concatenated'] + ' ' + ' '.join(issue['cleaned_comments'])
        
        for topic_id, words in topics_words.items():
            if issue_discusses_topic(issue_text, words):
                if decision_type == 'Manual':
                    manual_counts[topic_id] += 1
                else:
                    automatic_counts[topic_id] += 1

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Topic': list(topics_words.keys()),
    'Manual': [manual_counts[topic_id] for topic_id in topics_words.keys()],
    'Automatic': [automatic_counts[topic_id] for topic_id in topics_words.keys()]
})

# Plotting the chart
ax = plot_data.set_index('Topic').plot(kind='bar', stacked=True)
plt.title('Number of Issues per Topic')
plt.xlabel('Topic ID')
plt.ylabel('Number of Issues')

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.), 
                    ha='center', va='center', xytext=(0, 0), 
                    textcoords='offset points')

plt.show()
