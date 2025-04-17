import json
import shlex
import subprocess
import pandas as pd

file_path = './datasets/issues/Issues_2nd.xlsx'
df = pd.read_excel(file_path, sheet_name='HDFS')
base_url = 'https://issues.apache.org/jira/rest/api/2/issue/' 

def get(url):
    print(f"Fetching data from: {url}")
    info = subprocess.run(shlex.split(f'curl -s {url}'), stdout=subprocess.PIPE)
    return json.loads(info.stdout.decode())
def fetch_issue_details(issue_id):
    # Assuming this function retrieves issue details given an issue ID
    return  get(base_url+issue_id)


def main():
    # List to hold all the issue data
    issues_data = []

    # Iterate over each issue key in the DataFrame
    for issue_key in df['Issue key']:  # Assuming the column name is 'Issue Key'
        # URL for the Jira REST API endpoint for the issue
        api_url = f'https://issues.apache.org/jira/rest/api/2/issue/{issue_key}'

        issue_data = get(api_url)
        
        if 'fields' in issue_data:
            parentsummary = {}
            if 'parent' in issue_data['fields']:
                parent_id = issue_data['fields']['parent']['key']
                parent_data = get(base_url+parent_id)
            
                if parent_data:
                   
                    parentsummary ={
                    "issue_id": parent_data['key'],
                    "summary": parent_data['fields']['summary'],
                    "description": parent_data['fields'].get('description', 'No description available.'),
                    "status_name": parent_data['fields']['status']['name'],
                    "comment_count": parent_data['fields']['comment']['total'],
                    "comments": [comment['body'] for comment in parent_data['fields']['comment']['comments']],
                    "issue_type": parent_data['fields']['issuetype']['name'],
                    "n_attachement" : len(parent_data['fields'].get('attachment',[])),

                    }
                    print("Parent data fetched:",parent_data)

            readable_output = {
                "issue_id": issue_data['key'],
                "summary": issue_data['fields']['summary'],
                "description": issue_data['fields'].get('description', 'No description available.'),
                "status_name": issue_data['fields']['status']['name'],
                "comment_count": issue_data['fields']['comment']['total'],
                "comments": [comment['body'] for comment in issue_data['fields']['comment']['comments']],
                "issue_type": issue_data['fields']['issuetype']['name'],
                "n_attachement" : len(issue_data['fields'].get('attachment',[])),
                "parent": parentsummary

            }
        
            # Append the issue data to the list
            issues_data.append(readable_output)

    # Save all the issue data to a JSON file
    with open('./datasets/issues/issue_details/issues_data.json', 'w') as json_file:
        json.dump(issues_data, json_file, indent=4)

if __name__ == "__main__":
    main()
