import json
from collections import Counter
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import multiprocessing
import logging

# Set up logging for gensim to monitor the process
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Correct file path with raw string notation or double backslashes
input_file_path = "datasets/pre_processed_data/cleaned_issues_data_weightage.json"
lda_output_file_path = "datasets/lda_results/lda_result_wightage_8.json"



def generate_lda_results(input_file_path, lda_output_file_path):
    try:
        # Load JSON data
        with open(input_file_path, 'r') as file:
            data = json.load(file)

        # Extract tokens from the pre-processed text
        text_data = []
        for item in data:
            tokens = item['summary_description_concatenated'].split()
            text_data.append(tokens)

        # Prepare data for LDA
        dictionary = corpora.Dictionary.load("utils/lda_model_data/dictionary/dictionary.dict")
        corpus = corpora.MmCorpus("utils/lda_model_data/corpus/corpus.mm")

        # Function to compute perplexity and coherence values for different numbers of topics
        def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
            coherence_values = []
            perplexity_values = []
            for num_topics in range(start, limit, step):
                model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                        id2word=dictionary,
                                                        num_topics=num_topics,
                                                        random_state=100,
                                                        update_every=1,
                                                        chunksize=100,
                                                        passes=10,
                                                        alpha=0.01,
                                                        per_word_topics=True)
                coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_values.append(coherence_model.get_coherence())
                perplexity_values.append(model.log_perplexity(corpus))
            return coherence_values, perplexity_values

        # Compute the coherence and perplexity values
        limit = 11
        start = 3
        step = 1
        coherence_values, perplexity_values = compute_coherence_values(dictionary, corpus, text_data, limit, start, step)

        # Save the perplexity and coherence values to a JSON file
        lda_result = {
            "num_topics": list(range(start, limit, step)),
            "coherence_values": coherence_values,
            "perplexity_values": perplexity_values
        }
        with open(lda_output_file_path, 'w') as json_file:
            json.dump(lda_result, json_file, indent=4)

        print("LDA results have been written to 'lda_result.json'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    generate_lda_results(input_file_path, lda_output_file_path)
