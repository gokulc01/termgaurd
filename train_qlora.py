#!/usr/bin/env python3
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def main():
    model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
    output_dir = "termguard-qwen-lora"

    print("Loading dataset...")
    dataset = load_dataset(
        "json", data_files="training/data/train.jsonl", split="train"
    )

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # Keeps forward pass fast and memory-efficient
    )

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for QLoRA
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    print("Initializing trainer...")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,  # Disables AMP and the problematic GradScaler
        bf16=False,
        logging_steps=10,
        optim="paged_adamw_8bit",
        save_strategy="no",
        dataset_text_field="text",
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    print("Starting QLoRA training...")
    trainer.train()

    print(f"Saving final LoRA adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
