from src.models.base_model import BaseReasoningModel
from src.utils.json_utils import extract_json, leaves

class QuestionProcessor:
    def __init__(self, reasoning_model: BaseReasoningModel):
        self.model = reasoning_model
        self.default_system_prompt = "You are a helpful assistant who answers questions truthfully to the best of your knowledge."

    def ask(self, prompt: str, system_prompt: str = None) -> str:
        if system_prompt is None:
            system_prompt = self.default_system_prompt
        return self.model.ask_model(prompt, system_prompt)

    def extract_subquestions(self, question: str) -> list:
        prompt = f"""Break down the following question into intermediate sub-questions to approach answering it. Provide a list of intermediate sub-questions in JSON format. Do not directly answer the question.

Question: {question}"""
        response = self.ask(prompt)
        return list(leaves(extract_json(response)))

    def extract_subsubquestions(self, question: str, topic: str) -> list:
        prompt = f"""You are researching the following topic. Break down the question into intermediate sub-questions. Provide a list in JSON format.

Topic: {topic}
Question: {question}"""
        response = self.ask(prompt)
        return list(leaves(extract_json(response)))

    def question_to_header(self, question: str, topic: str = None) -> str:
        if topic:
            prompt = f"""Rewrite the following question with given context as a header title. Be concise. Respond in JSON format.

Context: {topic}
Question: {question}"""
        else:
            prompt = f"""Rewrite the following question as a header title. Be concise. Respond in JSON format.

Question: {question}"""
        response = self.ask(prompt)
        return list(leaves(extract_json(response)))[0]
