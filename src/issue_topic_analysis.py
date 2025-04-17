import json
from collections import Counter

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

issue_data_path = r'datasets\pre_processed_data\cleaned_issues_data.json'
topics_data_path = r'datasets\lda_results\weightage_words\lda_result_iteration2_weightage_8.json'

issue_data = read_json(issue_data_path)
topics_data = read_json(topics_data_path)

topics = {topic['topic_id']: topic['words'] for topic in topics_data}

def find_relevant_topics(text, topics):
    topic_scores = Counter()
    for topic_id, words in topics.items():
        words_list = words.split(' + ')
        for word in words_list:
            score, keyword = word.split('*')
            keyword = keyword.strip('"')
            if keyword in text:
                topic_scores[topic_id] += float(score)
    return topic_scores

issue_analysis = []

for issue in issue_data:
    summary_description = issue['summary_description_concatenated']
    comments = ' '.join(issue['cleaned_comments'])
    full_text = f"{summary_description} {comments}"
    
    topic_scores = find_relevant_topics(full_text, topics)
    
    top_topics = topic_scores.most_common(3)
    
    issue_analysis.append({
        'issue_id': issue['issue_id'],
        'status_name': issue['status_name'],
        'issue_type': issue['metadata']['issue_type'],
        'n_comments': issue['metadata']['n_comments'],
        'total_comment_length': issue['metadata']['total_comment_length'],
        'n_attachment': issue['metadata']['n_attachment'],
        'top_topics': [{'topic_id': topic_id, 'score': score} for topic_id, score in top_topics]
    })

output_path = 'datasets/issue_topic_analysis/issue_analysis_results.json'
with open(output_path, 'w') as output_file:
    json.dump(issue_analysis, output_file, indent=4)

for issue in issue_analysis:
    print(f"Issue ID: {issue['issue_id']}")
    print("Top 3 Topics with Scores:")
    for topic in issue['top_topics']:
        print(f"Topic ID: {topic['topic_id']}, Score: {topic['score']}")
    print() 

print(f"Analysis results saved to {output_path}")
