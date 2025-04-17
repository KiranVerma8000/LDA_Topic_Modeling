import json
from collections import Counter
import os
import gensim
from gensim import corpora
import logging


# Set up logging for gensim to monitor the process
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Correct file path with raw string notation or double backslashes
input_file_path = "datasets/pre_processed_data/cleaned_issues_data_weightage.json"
output_file_path = "datasets/vocabulary/vocabulary_result_iteration2_weightage_8.json"
lda_output_file_path = "datasets/lda_results/lda_result_iteration2_wightage_8.json"

output_model_path = "utils/lda_model_data/model/lda_model"
dictionary_path = "utils/lda_model_data/dictionary/dictionary.dict"
corpus_path = "utils/lda_model_data/corpus/corpus.mm"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Define stopwords (if not already removed during preprocessing)
stopwords = {'the', 'is', 'this', 'for', 'goes', 'here', 'another'}

try:
    # Load JSON data
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    # Extract tokens from the pre-processed text
    all_tokens = []
    text_data = []  # To store the token lists for each document
    for item in data:
        tokens = item['summary_description_concatenated'].split()
        all_tokens.extend(tokens)
        text_data.append(tokens)

    # Count the frequency of each token
    token_counts = Counter(all_tokens)

    # Create the vocabulary
    vocabulary = dict(token_counts)

    # Sort the vocabulary by frequency (optional)
    sorted_vocabulary = dict(sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))

    # Find tokens with the most frequency
    max_frequency = max(token_counts.values())
    most_frequent_tokens = [word for word, freq in token_counts.items() if freq == max_frequency]

    # Identify candidates for pre-processing (stopwords or numbers)
    candidates_for_preprocessing = [
        word for word in token_counts 
        if word in stopwords or word.isdigit()
    ]

    # Create the result dictionary
    result = {
        "vocabulary": sorted_vocabulary,
        "most_frequent_tokens": most_frequent_tokens,
        "candidates_for_preprocessing": candidates_for_preprocessing
    }

    # Write the result to a JSON file
    with open(output_file_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)

    print("Vocabulary and frequency data have been written to 'vocabulary_result.json'")

    # Prepare data for LDA
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    # Run LDA with standard parameters
    lda_model = gensim.models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15, alpha=0.01)

    dictionary.save(dictionary_path)
    corpora.MmCorpus.serialize(corpus_path, corpus)
    lda_model.save(output_model_path)
    # Extract topics
    topics = lda_model.print_topics(num_words=8)
    lda_topics = [{"topic_id": i, "words": topic} for i, topic in topics]

    # Write LDA topics to a JSON file
    with open(lda_output_file_path, 'w') as lda_file:
        json.dump(lda_topics, lda_file, indent=4)

    print("LDA results have been written to 'lda_topics_result.json'")

except Exception as e:
    print(f"An error occurred: {e}")


