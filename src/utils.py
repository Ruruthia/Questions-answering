import csv
from collections import defaultdict

import editdistance
from transformers import PreTrainedTokenizer


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


def modify_prompt(raw_prompt: str, history: list[str], tokenizer: PreTrainedTokenizer) -> str:

    def crop_history() -> list[str]:
        cropped_history = []
        cropped_history_token_len = 0
        prompt_token_len = len(tokenizer.encode(raw_prompt))
        i = 1
        while i <= len(history):
            cropped_history_token_len += len(tokenizer.encode("</s> <s>" + history[-i]))
            if cropped_history_token_len >= 127 - prompt_token_len:
                break
            cropped_history.append(history[-i])
            i += 1
        cropped_history.reverse()
        return cropped_history

    if len(history):
        return "</s> <s>".join(crop_history() + [raw_prompt])
    else:
        return raw_prompt


def modify_response(response: str) -> str:
    # Leave only the first line
    response = response.partition('\n')[0]
    # Remove everything after the last '.', '!', or '?'.
    last_sentence_end_index = max(response.rfind('.'), response.rfind('?'), response.rfind('!'))
    if last_sentence_end_index != -1:  # -1 means none were found
        response = response[:last_sentence_end_index + 1]
    return response.strip()
