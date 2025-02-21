from abc import ABC, abstractmethod

class BaseReasoningModel(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response based on the prompt and an optional system prompt."""
        pass