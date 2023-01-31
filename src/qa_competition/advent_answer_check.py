from pathlib import Path

import editdistance

rn = ['ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii',
      'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx', 'xxi', 'xxii']

rome_numbers = dict(zip(rn, range(2, 23)))


def numbers_from(s: str) -> set:
    res = set()
    for w in s.split():
        w = w.lower()
        if w.isdecimal():
            res.add(w)
        if w in rome_numbers:
            res.add(rome_numbers[w])
    return res


def is_number(s: str) -> bool:
    return s.lower() in rome_numbers or s.isdecimal()


def scaled_editdist(ans: str, cor: str) -> float:
    ans = ans.lower()
    cor = cor.lower()

    return editdistance.eval(ans, cor) / len(cor)


def single_match(a: str, c: str) -> bool:
    numbers_c = numbers_from(c)
    numbers_a = numbers_from(a)

    return numbers_a == numbers_c and scaled_editdist(a, c) < 0.5


def match(ans: str, cor: str) -> bool:
    return any(single_match(ans, c) for c in cor)


found_answers = []
correct_answers = []

for x in open(Path(__file__).parents[2] / 'data' / 'qa_competition' / 'correct_answers.txt'):
    x = x.strip()
    correct_answers.append(x.lower().split('\t'))

for x in open(Path(__file__).parents[2] / 'data' / 'qa_competition' / 'answers.txt'):
    x = x.strip()
    found_answers.append(x.lower())

N = len(correct_answers)
score = 0.0

for ans, cor in zip(found_answers, correct_answers):
    print(ans, cor)
    if match(ans, cor):
        score += 1

print('TOTAL SCORE:', score / len(correct_answers))
