import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
from tqdm import tqdm
import csv
import time

"""
Paper Goal & Intuition:
This script fine-tunes multiple LLMs on various subset sizes of the MMLU dataset to analyze
scaling laws in fine-tuning efficiency. The models are trained on progressively larger subsets
and then evaluated on a separate 500-example test set. The results are logged live.

Key Considerations:
- Uses LoRA for memory efficiency (can be toggled off for full fine-tuning).
- Evaluates on a fixed 500-example test set.
- Uses tqdm for progress tracking.
- Logs training/evaluation results in real-time for analysis.
- Runs a baseline evaluation with zero training for comparison.
"""

# Model list for evaluation
MODEL_LIST = [
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "HuggingFaceTB/SmolLM-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "tiiuae/Falcon3-1B-Instruct",
    "tiiuae/Falcon3-3B-Instruct",
    "tiiuae/Falcon3-7B-Instruct",
    "tiiuae/Falcon3-10B-Instruct",
]

# Training subset sizes for evaluation, including a 0-shot baseline
TRAIN_SUBSET_SIZES = [0, 100, 250, 500, 750, 1000]

# Load dataset
data = load_dataset("cais/mmlu", "professional_law")

def compute_accuracy(trainer, eval_dataset, tokenizer, model):
    """Computes accuracy using multiple-choice log-likelihood ranking."""
    correct, total = 0, 0
    for example in tqdm(eval_dataset, desc="Evaluating", leave=False):
        question = example["question"]
        choices = example["choices"]
        correct_answer = example["answer"]
        
        best_choice = None
        best_score = float("inf")
        
        for i, choice in enumerate(choices):
            prompt = f"Question: {question} Answer: {choice}"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
            
            if loss < best_score:
                best_score = loss
                best_choice = i
        
        print(f"Predicted: {best_choice}, Correct: {correct_answer}")
        if best_choice == correct_answer:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    print(f"Final Accuracy: {accuracy:.3f}")
    return accuracy

# Iterate through models and subset sizes
for model_name in tqdm(MODEL_LIST, desc="Models"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    
    # Tokenizer function
    def tokenize_function(examples):
        tokenized = tokenizer(examples["question"], truncation=True, padding="max_length", max_length=2048)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_data = data.map(tokenize_function, batched=True)
    
    # 8-bit Quantization for large models
    bnb_config = BitsAndBytesConfig(load_in_8bit=True) if "7B" in model_name else None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    use_lora = True
    if use_lora:
        lora_config = LoraConfig(
            r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # CSV setup
    results_file = "mmlu_finetune_results.csv"
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "subset_size", "eval_loss", "eval_time", "accuracy"])
    
    for subset_size in tqdm(TRAIN_SUBSET_SIZES, desc=f"Training Subsets for {model_name}"):
        train_subset = tokenized_data["test"].select(range(min(subset_size, len(tokenized_data["test"])))) if subset_size > 0 else []
        eval_set = tokenized_data["test"].select(range(500))
        
        train_args = TrainingArguments(
            output_dir=f"./checkpoints/{model_name}_{subset_size}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=4 if "7B" in model_name else 16,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4 if "7B" in model_name else 2,
            fp16=torch.cuda.is_bf16_supported() == False,
            bf16=torch.cuda.is_bf16_supported(),
            num_train_epochs=3,
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            optim="adamw_torch_fused",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_subset if subset_size > 0 else None,
            eval_dataset=eval_set,
            tokenizer=tokenizer,
        )

        if subset_size > 0:
            print(f"Starting training for {model_name} on {subset_size} examples...")
            try:
                trainer.train()
            except ValueError as e:
                print(f"Error during training: {e}")
                continue
        
        print("Evaluating...")
        start_time = time.time()
        metrics = trainer.evaluate(eval_dataset=eval_set)
        eval_time = time.time() - start_time
        accuracy = compute_accuracy(trainer, eval_set, tokenizer, model)
        
        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([model_name, subset_size, metrics.get("eval_loss", "N/A"), eval_time, accuracy])

        print("Training & evaluation complete. Results saved.")

print("All models processed successfully.")
