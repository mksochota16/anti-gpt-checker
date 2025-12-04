import pickle
from typing import List, Tuple, Dict


class SimpleLanguageStatistics:
    language_name: str
    overall_counts: Dict[str, int] = {}
    word_probabilities: Dict[str, float] = {}
    _total_words: int = 0
    def __init__(self, language_name: str):
        self.language_name = language_name


    def add_texts(self, texts: List[str]) -> None:
        """
        Add new texts to the model and update word counts.

        Parameters:
        - texts (List[str]): List of texts.
        """
        for text in texts:
            if text is None:
                continue
            words = text.split()
            self._total_words += len(words)
            for word in words:
                self.overall_counts[word] = self.overall_counts.get(word, 0) + 1


    def compute_probabilities(self) -> Dict[str, float]:
        """
        Compute the language model based on current word counts.

        Returns:
        - Tuple[Dict[str, int], Dict[str, float]]: Tuple containing overall word counts and language model.
        """
        self.word_probabilities = {word: count / self._total_words for word, count in self.overall_counts.items()}
        return self.word_probabilities

    def save_to_file(self, file_path: str) -> None:
        """
        Save the current state of the instance to a file.

        Parameters:
        - file_path (str): The path to the file where the object should be saved.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_file(file_path: str) -> 'SimpleLanguageStatistics':
        """
        Load a SimpleLanguageStatistics instance from a file.

        Parameters:
        - file_path (str): The path to the file from which to load the object.

        Returns:
        - SimpleLanguageStatistics: The loaded SimpleLanguageStatistics instance.
        """
        with open(file_path, 'rb') as file:
            instance = pickle.load(file)
        return instance