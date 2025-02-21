import transformers
from transformers import TextStreamer
from unsloth import FastLanguageModel
from src.models.base_model import BaseReasoningModel

class LanguageReasoningModel(BaseReasoningModel):
    def __init__(self, model_name: str, max_seq_length: int = 4048, load_in_4bit: bool = True, dtype=None):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)

    def generate(self, prompt: str, system_prompt: str = "You are a helpful assistant who answers questions truthfully to the best of your knowledge.") -> str:
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        results = self.model.generate(**inputs, streamer=streamer, max_new_tokens=4048)
        # Decode the result (this follows the original logic)
        return self.tokenizer.decode(token_ids=results.cpu().numpy().tolist()[0])
