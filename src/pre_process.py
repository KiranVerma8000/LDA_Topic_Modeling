import json
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Function to map NLTK POS tags to WordNet POS tags

with open("/Users/mitul/Documents/study/sem 4/DSSE/Assignmets/Assignment 2/ds4se2-group6/datasets/pre_processed_data/ontologies_weight_word_dictionary.json", 'r') as file:
    ontology_weights = json.load(file)
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN  # Default to noun if no tag matches

# Function to clean text
def clean_text(text):
    if isinstance(text, list):
        text = ' '.join(text)

    if text is None or not text.strip():
        return ""

    # Lowercase and remove unwanted characters
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'\{.*?\}', '', cleaned_text)
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)
    cleaned_text = re.sub(r'\(.*?\)', '', cleaned_text)
    cleaned_text = re.sub(r'<.*?>', '', cleaned_text)
    cleaned_text = re.sub(r'\|', '', cleaned_text)

    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(cleaned_text)
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]

    pos_tags = nltk.pos_tag(filtered_words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

    return check_words(' '.join(lemmatized_words),ontology_data)

# Function to check words against ontology
def check_words(words, ontology_data):
    result = []
    for word in words.split():
        found = False
        for ontology_class in ontology_data:
            if word in ontology_class["content"]:
                weight = int(ontology_weights.get(word, 1))
                for i in range(weight):
                    result.append(ontology_class["name"])
                found = True
                break
        if not found:
            if word.isdigit():
                result.append("NumberClass")
            else:
                result.append(word)
    return ' '.join(result)

# Read input files
with open(r'datasets\issues\issue_details\issues_data.json', 'r') as file:
    data_list = json.load(file)
with open(r"datasets\issues\pre_processed_data\ontologies.json", 'r') as file:
    ontology_data = json.load(file)

# Process data
cleaned_data_list = []

for data in data_list:
    cleaned_summary = clean_text(data["summary"])
    cleaned_description = clean_text(data["description"])
    summary_description_concatenated = cleaned_summary + " " + cleaned_description

    cleaned_comments = [clean_text(comment) for comment in data["comments"]]
    total_comments = len(cleaned_comments)
    
    total_comment_length = sum(len(comment) for comment in cleaned_comments)

    cleaned_data = {
        "issue_id": data["issue_id"],
        "cleaned_summary": cleaned_summary,
        "cleaned_description": cleaned_description,
        "summary_description_concatenated": summary_description_concatenated,
        "cleaned_comments": cleaned_comments,
        "status_name": data["status_name"],
        "metadata": {
            "n_comments": total_comments,
            "total_comment_length": total_comment_length,
            "issue_type": data["issue_type"],
            "n_attachment": data.get("n_attachement", 0),
            "Parent": {
                "issue_id": data.get("parent", {}).get("issue_id") or "",
                "cleaned_summary": clean_text(data.get("parent", {}).get("summary", "")),
                "cleaned_description": clean_text(data.get("parent", {}).get("description", "")),
                "status_name": data.get("parent", {}).get("status_name", ""),
                "metadata": {
                    "n_comments": len(data.get("parent", {}).get("comments", [])),
                    "total_comment_length": sum(len(comment) for comment in data.get("parent", {}).get("comments", [])),
                    "issue_type": data.get("parent", {}).get("issue_type", ""),
                    "n_attachment": data.get("parent", {}).get("n_attachement", 0)
                } if data.get("parent") else {}
            }
        }
    }

    cleaned_data_list.append(cleaned_data)

# Write cleaned data to output file
with open('datasets/issues/pre_processed_data/cleaned_issues_data.json', 'w') as outfile:
    json.dump(cleaned_data_list, outfile, indent=4)

print("Pre-processing complete. Cleaned data has been saved.")
