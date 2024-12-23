from abc import ABC, abstractmethod

class Completion(ABC):
    @abstractmethod
    def get_completion(self, context: str, question: str) -> str:
        """
        Generates a text completion for the given prompt.

        Args:
            self (Completion): The completion instance.
            context (str): The context to generate a completion for.
            question (str): The question to generate a completion for.

        Returns:
            str: The generated text completion.
        """
        pass
