import csv
from collections import defaultdict


def read_qa_tsv(questions_answers_path: str):
    with open(questions_answers_path, 'r') as in_file:
        csv_file = csv.reader(in_file, delimiter='\t')
        qa_dict = {line[0]: line[1:] for line in csv_file}
    return qa_dict


def read_definitions(definitions_path: str):
    definitions_dict = defaultdict(list)
    with open(definitions_path, 'r') as in_file:
        for line in in_file:
            word, definition = line.split('###')
            definitions_dict[word.strip()].append(definition.strip())
    return definitions_dict
