import json
import matplotlib.pyplot as plt

# Correct file path with raw string notation or double backslashes
lda_output_file_path = "/Users/mitul/Documents/study/sem 4/DSSE/Assignmets/Assignment 2/ds4se2-group6/datasets/lda_results/lda_result_wightage.json"

try:
    # Load LDA results from JSON file
    with open(lda_output_file_path, 'r') as file:
        lda_result = json.load(file)

    # Extract data for plotting
    num_topics = lda_result['num_topics']
    coherence_values = lda_result['coherence_values']
    perplexity_values = lda_result['perplexity_values']

    # Plot perplexity values
    plt.plot(num_topics, perplexity_values, label='Perplexity')
    plt.title("Perplexity and Coherence by Number of Topics")
    plt.xlabel("Number of Topics")
    plt.ylabel("Perplexity")
    
    # Plot coherence values
    plt.plot(num_topics, coherence_values, label='Coherence')
    
    plt.legend()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
