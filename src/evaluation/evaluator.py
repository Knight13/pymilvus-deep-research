import numpy as np
from bert_score import score as bert_score

class LanguageReasoningModelEvaluator:
    def __init__(self, language: str = "en"):
        self.language = language

    def evaluate_bertscore(self, predictions, references):
        # Compute BERTScore for a list of prediction and reference strings.
        P, R, F1 = bert_score(predictions, references, lang=self.language, verbose=True)
        return {
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(),
            "bert_score_f1": F1.mean().item()
        }

    def evaluate(self, predictions, references):
        bert_results = self.evaluate_bertscore(predictions, references)
        results = {**bert_results}
        return results
