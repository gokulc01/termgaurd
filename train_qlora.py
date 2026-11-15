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
        "json", data_files="data/termguard_dataset_200_items.json", split="train"
    )

    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

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

    # Map your custom JSON keys to Qwen's expected ChatML format
    def format_chat_template(example):
        user_content = (
            f"Task: {example['task_prompt']}\n"
            f"Allowed roots: {example['allowed_roots']}\n"
            f"Context: {example['rag_context']}\n"
            f"Feedback: {example['feedback']}"
        )

        messages = [
            {
                "role": "system",
                "content": "You are TermGuard-Agent, a safety-critical Linux shell copilot. You must output strictly valid JSON adhering to the provided StructuredAction schema.",
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["target_action"]},
        ]

        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return example

    print("Formatting dataset to ChatML...")
    dataset = dataset.map(format_chat_template)

    print("Initializing trainer...")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=150,  # Automatically stops after 150 steps. Increase if dataset is larger than 1200 rows.
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        save_strategy="no",
        dataset_text_field="text",
        max_length=1024,
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
