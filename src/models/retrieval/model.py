from abc import ABC, abstractmethod
from src.utils import read_definitions


class RetrievalModel(ABC):
    def __init__(self, definitions_path: str):
        self._definitions: list[str] = []
        self._names: list[str] = []
        self._index_definitions(read_definitions(definitions_path))

    def _get_probable_answers(self, question: str, max_answers: int = 10) -> list[int]:
        """Sparse retrieval"""
        pass

    @abstractmethod
    def _match_question(self, question: str) -> int:
        """
        Finds the index of definition that best matches the question.
        Args:
            question: Question to match the definitions to
            definitions: List of definitions.

        Returns:
            Index of a definition that best matches the question.
        """
        pass

    def _index_definitions(self, definitions_dict: dict[str, list[str]]):
        for name, defs in definitions_dict.items():
            for definition in defs:
                if definition is not None:
                    self._definitions.append(definition)
                    self._names.append(name)

    def answer_question(self, question: str, max_answers: int = 10) -> str:
        idx_probab = self._get_probable_answers(question, max_answers)
        idx = self._match_question(question, idx_probab)
        return self._names[idx]
