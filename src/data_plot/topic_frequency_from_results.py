import json
import matplotlib.pyplot as plt

# Load LDA topics results from JSON file
#lda_topics_file_path = r"datasets\issues\vocabulary\lda_topics_result_iteration1.json"
lda_topics_file_path = r"datasets\lda_results\lda_topics_result_iteration2.json"

try:
    # Load LDA topics results
    with open(lda_topics_file_path, 'r') as file:
        lda_topics_results = json.load(file)

    # Initialize a figure for plotting
    plt.figure(figsize=(15, 10))

    # Generate word frequency graphs for each topic
    for topic in lda_topics_results:
        topic_id = topic['topic_id']
        words = topic['words']
        
        # Extract words and probabilities from the string representation
        word_probs = [word_prob.split('*') for word_prob in words.split(' + ')]
        words, freqs = zip(*[(word.strip('"'), float(prob)) for prob, word in word_probs])

        # Plot word frequencies for the current topic
        plt.barh(words, freqs, label=f"Topic {topic_id}")

    # Add labels and legend
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.title("Word Frequencies for All Topics")
    plt.legend()
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
