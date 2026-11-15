# merge_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
adapter_path = "termguard-qwen-lora"
merged_path = "termguard-qwen-merged"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_name, torch_dtype=torch.float16, device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(base_name)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging weights (this takes a minute)...")
merged_model = model.merge_and_unload()

print(f"Saving merged model to {merged_path}...")
merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print("Done!")
