import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import os
import json

# --- 1. 環境設定與模型選擇 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# QLoRA 量化設定
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_compute_dtype=torch.bfloat16
)

# --- 2. 載入模型與 Tokenizer ---
model_path = "./Meta-Llama-3-8B-Instruct"
print(f"正在載入模型: {model_path}...")
# 第一次執行會自動下載。注意：Llama-3 不需要 trust_remote_code=True
model_path = "./Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Llama-3 tokenizer 沒有預設的 pad_token，我們將它設為 eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

print("模型和 Tokenizer 載入完成。")

# --- 3. 準備訓練資料
IGNORE_INDEX = -100 # 忽略指數，用於遮罩提示部分的損失
def preprocess_function_llama3(examples):
    prompts, responses = [], []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        # Llama-3 的官方對話格式
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a helpful assistant designed to analyze text and output structured JSON.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{instruction}\n\n文本內容：\n{input_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        prompts.append(prompt)
        # 確保答案以結束符號結尾
        responses.append(json.dumps(output, ensure_ascii=False) + tokenizer.eos_token)
    
    # 分別對提示和答案進行 tokenize
    tokenized_prompts = tokenizer(prompts, truncation=False, add_special_tokens=False)
    tokenized_responses = tokenizer(responses, truncation=False, add_special_tokens=False)

    # 創建 input_ids 和 labels
    all_input_ids, all_labels = [], []
    for prompt_ids, response_ids in zip(tokenized_prompts['input_ids'], tokenized_responses['input_ids']):
        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
        
        # 截斷長度
        max_length = 1024
        if len(input_ids) > max_length:
            input_ids, labels = input_ids[:max_length], labels[:max_length]
        
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        
    return {"input_ids": all_input_ids, "labels": all_labels}

print("正在使用 Llama-3 模板處理資料集...")
raw_dataset = load_dataset('json', data_files='同業(202206~202307).json', split="train")
tokenized_dataset = raw_dataset.map(preprocess_function_llama3, batched=True, remove_columns=raw_dataset.column_names)
print("資料集處理完成。")

# --- 4. LoRA 與訓練參數設定 ---
model = prepare_model_for_kbit_training(model)
# Llama-3 的可訓練模塊比 Qwen2 更多一些
peft_config = LoraConfig(
    r=64, 
    lora_alpha=64, 
    lora_dropout=0.1, 
    bias="none", 
    task_type="CAUSAL_LM", 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# 訓練參數
training_args = TrainingArguments(
    output_dir="./llama3-8b-competitor-finetune",
    num_train_epochs=3, 
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4,
    learning_rate=2e-5, 
    fp16=True, 
    logging_steps=3, 
    save_strategy="epoch",
    max_grad_norm=0.3, 
    weight_decay=0.01, 
    lr_scheduler_type='cosine', 
    warmup_ratio=0.03
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# --- 5. 開始訓練 ---
print("一切準備就緒，使用 Llama-3 基礎模型開始微調...")
resume_from_checkpoint = True # 設為 True 來啟用斷點續訓
# 或者 resume_from_checkpoint = "path/to/your/checkpoint-xxx"

print("準備開始訓練...")
# 檢查是否需要從斷點恢復
if resume_from_checkpoint:
    print(f"將從最近的檢查點恢復訓練，工作目錄: {training_args.output_dir}")
    trainer.train(resume_from_checkpoint=True)
else:
    print("將開始一次全新的訓練...")
    trainer.train()
    
print("訓練完成！")

# --- 6. 保存最終模型 ---
final_model_path = "./llama3-8b-competitor-final"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"模型已成功保存至 {final_model_path}")
