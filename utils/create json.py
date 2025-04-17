
import json
with open("/Users/mitul/Documents/study/sem 4/DSSE/Assignmets/Assignment 2/ds4se2-group6/datasets/pre_processed_data/ontologies_weightage.json") as file:
    ontology_data = json.load(file)

ontoolgoy_final = []
ontology_weights = {}
for ontology in ontology_data:
    for word in ontology['content']:
        ontology_weights[word] = ontology['weight']
     

with open("/Users/mitul/Documents/study/sem 4/DSSE/Assignmets/Assignment 2/ds4se2-group6/datasets/pre_processed_data/ontologies_weight_word_dictionary.json", 'w') as json_file:
            json.dump(ontology_weights, json_file, indent=4)


