import csv


def read_qa_tsv(questions_answers_path: str):
    with open(questions_answers_path, 'r') as in_file:
        csv_file = csv.reader(in_file, delimiter='\t')
        qa_dict = {line[0]: line[1:] for line in csv_file}
    return qa_dict
