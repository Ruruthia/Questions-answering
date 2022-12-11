from abc import ABC, abstractmethod
from src.utils import read_definitions


class RetrievalModel(ABC):
    def __init__(self, definitions_path: str):
        self._definitions: list[str] = []
        self._names: list[str] = []
        self._index_definitions(read_definitions(definitions_path))

    @abstractmethod
    def _match_question(self, question: str, definitions: list[str]) -> int:
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
                self._definitions.append(definition)
                self._names.append(name)

    def answer_question(self, question: str) -> str:
        idx = self._match_question(question, self._definitions)
        return self._names[idx]

