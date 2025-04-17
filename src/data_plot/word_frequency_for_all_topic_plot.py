import json
import matplotlib.pyplot as plt
import gensim
from gensim import corpora

# Correct file paths with raw string notation or double backslashes
lda_output_file_path = r"datasets\issues\vocabulary\lda_result.json"
input_file_path = r"datasets\issues\pre_processed_data\cleaned_issues_data.json"

try:
    optimal_num_topics = 10  # Update with the optimal number of topics
    dictionary = corpora.Dictionary.load(r"datasets\LDA_model_data\dictionary\dictionary.dict")
    corpus = corpora.MmCorpus(r"datasets\LDA_model_data\corpus\corpus.mm")
    lda_model = gensim.models.LdaModel.load(r"datasets\LDA_model_data\model\lda_model")

    # Initialize a figure for plotting
    fig, axes = plt.subplots(5, 2, figsize=(20, 25), constrained_layout=True)  # 5 rows, 2 columns for 10 topics
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Generate word frequency graphs for each topic
    for topic_id in range(optimal_num_topics):
        # Extract top words and probabilities for the current topic
        topic_words = lda_model.show_topic(topic_id, topn=10)  # Get top 10 words for the topic
        words, freqs = zip(*topic_words)

        # Plot word frequencies
        ax = axes[topic_id]
        ax.bar(words, freqs, color='skyblue')
        ax.set_title(f"Word Frequencies for Topic {topic_id + 1}")
        ax.set_xlabel("Words")
        ax.set_ylabel("Frequency")
        ax.set_xticklabels(words, rotation=45, ha='right')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect parameter to fit x-axis labels better
    plt.suptitle("Word Frequencies for All Topics", y=1.02)
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
