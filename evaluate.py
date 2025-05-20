from datasets import load_dataset, DownloadConfig
from src.evaluation.evaluator import LanguageReasoningModelEvaluator
from src.models.language_model import LanguageReasoningModel
import torch
from dynaconf import Dynaconf
import fsspec
fsspec.config.conf["default_timeout"] = 6000

def run_evaluation(cfg: object):
    # Load HotPotQA. Note: the original test set may not include ground truth answers.
    # If so, consider using the "dev" split.
    download_config = DownloadConfig(max_retries=10, resume_download=True, storage_options={"timeout": 6000})
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True, download_config=download_config, download_mode="reuse_dataset_if_exists")
    if cfg.num_samples in list(range(1, 7401)):
        dataset = dataset.select(range(cfg.num_samples))
    
    # Initialize your reasoning model.
    model_name = cfg.model_name
    reasoning_model = LanguageReasoningModel(
        model_name=model_name,
        max_seq_length=4048,
        load_in_4bit=True,
        dtype=None
    )
    
    # Set the model to evaluation mode and (if using GPU) ensure it is on the correct device.
    reasoning_model.model.eval()
    if torch.cuda.is_available():
        reasoning_model.model.to("cuda")
    
    predictions = []
    references = []
    
    print(f"Evaluating on {len(dataset)} samples from HotPotQA...\n")
    
    for idx, sample in enumerate(dataset):
        question = sample["question"]
        # Ground truth answer field may vary (e.g., "answer" or "answer_text")
        ref = sample["answer"]
        
        # Generate the prediction (this call is synchronous; consider batching if needed).
        pred = reasoning_model.ask_model(question)
        
        predictions.append(pred)
        references.append(ref)
        
        print(f"Sample {idx+1}:")
        print(f"Question: {question}")
        print(f"Reference: {ref}")
        print(f"Prediction: {pred}")
        print("-" * 50)
    
    # Initialize evaluator and compute BERTScore metrics.
    evaluator = LanguageReasoningModelEvaluator(language="en")
    bert_metrics = evaluator.evaluate_bertscore(predictions, references)
    
    print("\nBERTScore Metrics on HotPotQA Evaluation:")
    print(f"Precision: {bert_metrics['bert_score_precision']:.4f}")
    print(f"Recall:    {bert_metrics['bert_score_recall']:.4f}")
    print(f"F1:        {bert_metrics['bert_score_f1']:.4f}")

config_path = "./configs/eval.yaml"
if __name__ == "__main__":
    cfg = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[config_path],
    )
    
    run_evaluation(cfg)
