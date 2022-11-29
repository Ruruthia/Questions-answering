from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass
from collections import defaultdict as dd
import re
from src.models.word2vec import Word2Vec
import numpy as np
from scipy import spatial
import pandas as pd

@dataclass
class AnsweredQuestion:
    question: str
    answers: List[str]


class QuestionsClusterer(ABC):
    # @abstractmethod
    def _cluster_questions(
            self,
            questions_answers_path: str,
    ) -> None:
        """
        Args:
            questions_answers_path: Path to file containing the questions and answers to be clustered.

        Returns:
            SUBJECT TO CHANGE.
            None. Initializes the QuestionClusterer class with some kind of mapping between questions and clusters.
            Alternatively returns the mapping?
            Note: it would be better to have a mapping from cluster_id to questions from this cluster,
            not other way round.
        """
        with open(f"{questions_answers_path}/expected.tsv", 'r') as f:
            answers = f.readlines()
            answers = [a.strip().split("\t") for a in answers]

        with open(f"{questions_answers_path}/in.tsv", 'r') as f:
            questions = f.readlines()

        QA_data = [{'question': questions[i], 'answers': answers[i]} for i in range(len(answers))]

        self.clusters = dd(list)

        for d in QA_data:
            if re.match("Czy ", d['question']) and not re.match(r"Czy .* czy", d['question']):
                self.clusters["czy"].append(AnsweredQuestion(d['question'], d['answers']))
            elif re.match("któr", d['question'].split(" ")[1]) or re.match("jak", d['question'].split(" ")[1]):
                self.clusters[d['question'].split(" ")[2]].append(AnsweredQuestion(d['question'], d['answers']))
            elif re.match("któr", " ".join(d['question'].split(" ")[:2]).lower()):
                self.clusters[d['question'].split(" ")[1]].append(AnsweredQuestion(d['question'], d['answers']))
            elif re.match("któr", " ".join(d['question'].split(" ")[:3]).lower()):
                self.clusters[d['question'].split(" ")[2]].append(AnsweredQuestion(d['question'], d['answers']))
            elif re.match("któr", " ".join(d['question'].split(" ")[:4]).lower()):
                self.clusters[d['question'].split(" ")[3]].append(AnsweredQuestion(d['question'], d['answers']))
            elif re.match("jak.* nazyw.* się", " ".join(d['question'].split(" ")[:2]).lower()):
                self.clusters[d['question'].split(" ")[3]].append(AnsweredQuestion(d['question'], d['answers']))
            elif re.match(r"jak[a-z]* (?!nazyw)", " ".join(d['question'].split(" ")[:2]).lower()):
                self.clusters[d['question'].split(" ")[1]].append(AnsweredQuestion(d['question'], d['answers']))
            elif re.match(r".* czy ", d['question']):
                self.clusters['options'].append(AnsweredQuestion(d['question'], d['answers']))
            elif re.match("Ile ", d['question']):
                self.clusters['counters'].append(AnsweredQuestion(d['question'], d['answers']))
            elif re.match("Kto ", d['question']) or re.match(".* kto ", d['question']) or re.match("Kogo ", d['question']) or re.match(".* kto ", d['question']):
                self.clusters['who'].append(AnsweredQuestion(d['question'], d['answers']))
            else:
                self.clusters['other'].append(AnsweredQuestion(d['question'], d['answers']))

        self.w2v = Word2Vec()

    # @abstractmethod
    def cluster_single_question(
            self,
            question: str
    ) -> int:
        """
        Used for the questions from testing set (that we want to answer).

        Args:
            question: Text of the question to be clustered

        Returns:
            cluster_id: int, identificator of the cluster that the question belongs to.
        """
        if re.match("Czy ", question) and not re.match(r"Czy .* czy", question):
            cluster = "czy"
        elif re.match("któr", question.split(" ")[1]) or re.match("jak", question.split(" ")[1]):
            cluster = question.split(" ")[2]
        elif re.match("któr", " ".join(question.split(" ")[:2]).lower()):
            cluster = question.split(" ")[1]
        elif re.match("któr", " ".join(question.split(" ")[:3]).lower()):
            cluster = question.split(" ")[2]
        elif re.match("któr", " ".join(question.split(" ")[:4]).lower()):
            cluster = question.split(" ")[3]
        elif re.match("jak.* nazyw.* się", " ".join(question.split(" ")[:2]).lower()):
            cluster = question.split(" ")[3]
        elif re.match(r"jak[a-z]* (?!nazyw)", " ".join(question.split(" ")[:2]).lower()):
            cluster = question.split(" ")[1]
        elif re.match(r".* czy ", question):
            cluster = 'options'
        elif re.match("Ile ", question):
            cluster = 'counters'
        elif re.match("Kto ", question) or re.match(".* kto ", question) or re.match("Kogo ", question) or re.match(".* kto ", question):
            cluster = 'who'
        else:
            cluster = 'other'
        return cluster

    # @abstractmethod
    def sample_questions_from_cluster(
            self,
            cluster_id: int,
            num_questions_to_get: Optional[int] = 5,
            question: Optional[str] = None

    ) -> List[AnsweredQuestion]:
        """

        Args:
            cluster_id: Identificator of the cluster that we want to sample from
            num_questions_to_get: Number of samples to return.

        Returns: List of maximum length equal to num_questions_to_get, of AnsweredQuestions.

        """
        if cluster_id not in self.clusters.keys():
            cluster_id = 'other'
        if question is not None:
            emb = self.w2v.get_embedding(question)
            cosines = [1 - spatial.distance.cosine(emb, self.w2v.get_embedding(d.question)) for d in self.clusters[cluster_id]]
            similar_questions = pd.Series(cosines).abs().sort_values().iloc[:num_questions_to_get].index.values.tolist()
            return [self.clusters[cluster_id][i] for i in similar_questions]
        else:
            return [self.clusters[cluster_id][i] for i in np.random.choice(np.arange(len(self.clusters[cluster_id])),  num_questions_to_get)]
