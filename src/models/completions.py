from abc import ABC, abstractmethod

class Completion(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """
        Generates a text completion for the given prompt.

        Args:
            prompt (str): The input text prompt to generate a completion for.

        Returns:
            str: The generated text completion.
        """
        pass