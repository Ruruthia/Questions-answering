from abc import ABC, abstractmethod
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class AnsweredQuestion:
    question: str
    answers: List[str]


class QuestionsClusterer(ABC):
    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def sample_questions_from_cluster(
            self,
            cluster_id: int,
            num_questions_to_get: Optional[int] = 5
    ) -> List[AnsweredQuestion]:
        """

        Args:
            cluster_id: Identificator of the cluster that we want to sample from
            num_questions_to_get: Number of samples to return.

        Returns: List of maximum length equal to num_questions_to_get, of AnsweredQuestions.

        """