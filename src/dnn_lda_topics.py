import pandas as pd
import re
import json
from collections import Counter
from scipy.stats import chi2_contingency

# Load the Excel file
file_path = './datasets/Issues_2nd.xlsx'
df = pd.read_excel(file_path, sheet_name='HDFS')

# Load the topics from the JSON file
topics_data_path = r'datasets/lda_results/weightage_words/lda_result_iteration2_weightage_8.json'
with open(topics_data_path, 'r') as file:
    topics = json.load(file)

# Load the issues data from the JSON file
issues_data_path = 'datasets/pre_processed_data/cleaned_issues_data.json'
with open(issues_data_path, 'r') as file:
    issues_data = json.load(file)

# Extract top 6 words from topics
def extract_top_words(topic, n=6):
    words_with_weights = re.findall(r'(\d+\.\d+)\*\"([^\"]*)\"', topic["words"])
    sorted_words = sorted(words_with_weights, key=lambda x: float(x[0]), reverse=True)
    return [word for _, word in sorted_words[:n]]

topics_words = {topic["topic_id"]: extract_top_words(topic) for topic in topics}

# Function to check if an issue discusses a topic
def issue_discusses_topic(issue_text, topic_words):
    return any(word in issue_text for word in topic_words)

# Initialize a contingency table
design_decisions = ['Existence', 'Property', 'Executive']
contingency_table = {topic_id: {decision: 0 for decision in design_decisions} for topic_id in topics_words.keys()}

# Fill the contingency table
for _, row in df.iterrows():
    decisions = row['Types of decision'].split()
    decision_types = {design_decisions[i]: (decisions[i] == 'True') for i in range(len(decisions))}
    issue_id = row['Issue key']
    issue = next((issue for issue in issues_data if issue['issue_id'] == issue_id), None)
    
    if issue:
        issue_text = issue['summary_description_concatenated']
        
        for topic_id, words in topics_words.items():
            if issue_discusses_topic(issue_text, words):
                for decision_type, is_true in decision_types.items():
                    if is_true:
                        contingency_table[topic_id][decision_type] += 1

# Convert contingency table to a DataFrame
contingency_df = pd.DataFrame(contingency_table).T
print(contingency_df)

# Perform the chi-square test for each type of decision
for decision_type in design_decisions:
    table = contingency_df[[decision_type]]
    chi2, p, _, _ = chi2_contingency(table)
    print(f'Chi2 Statistic for {decision_type}: {chi2}')
    print(f'p-value for {decision_type}: {p}')

    if p < 0.05:
        print(f"There is a significant association between {decision_type} and LDA topics.")
    else:
        print(f"There is no significant association between {decision_type} and LDA topics.")
