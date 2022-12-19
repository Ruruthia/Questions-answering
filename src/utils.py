import csv
from collections import defaultdict

import editdistance


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


def scaled_editdist(ans: str, cor: str) -> float:
    ans = ans.lower()
    cor = cor.lower()

    return editdistance.eval(ans, cor) / len(cor)


def single_match(a: str, c: str) -> bool:
    if c.isdecimal():
        return a == c
    return scaled_editdist(a, c) < 0.5


def match(ans: str, cor: list[str]) -> bool:
    return any(single_match(ans, c) for c in cor)
