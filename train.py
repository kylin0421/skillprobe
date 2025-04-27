import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
import transformers
from peft import get_peft_model, PrefixTuningConfig, TaskType,  PeftModel, PeftConfig
import json
import random
import wandb

import os
from evaluate import evaluate
from utils.dataloader import SkillProbingDataset

# 2. 自定义 Trainer（重写 compute_loss）
class CustomTrainer(Trainer):
    '''
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    '''

# 3. 加载 TinyLlama模型和Tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 必须设 pad_token
tokenizer.pad_token = tokenizer.eos_token

# 4. PrefixTuning配置
peft_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    num_virtual_tokens=20,   # prefix长度
    prefix_projection=True,  # 加projection增强能力
)

# 5. 封装PEFT模型
model = get_peft_model(base_model, peft_config)

# 6. 加载SkillProbing Dataset
train_dataset = SkillProbingDataset("./data/reverse_letter_train.json", tokenizer)
valid_dataset = SkillProbingDataset("./data/reverse_letter_valid.json", tokenizer)


# 7. 设置训练参数

out_dir="./out/prefix_reverse_letter"

training_args = TrainingArguments(
    output_dir=out_dir,          # 保存模型的位置
    eval_strategy="steps",
    eval_steps= 200,           
    save_strategy="best",   
    logging_strategy="steps",               # logging
    logging_steps=30,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=50,
    learning_rate=5e-4,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=True,                              # 如果显卡支持，可以加速
    report_to="none",                 
    load_best_model_at_end=False,   
    metric_for_best_model="eval_loss",  # 根据哪个metric判断
    greater_is_better=False,  
)

# 8. 初始化 Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3)  
    ]
)

# 9. 开始训练！

#trainer.train()

def find_latest_checkpoint(output_dir):
    checkpoint_dirs = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoint_dirs:
        raise ValueError("No checkpoint found in output_dir")
    # 根据数字排序，找最大的
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
    return latest_checkpoint


best_checkpoint_path = find_latest_checkpoint(out_dir)

print(f"Loading model from: {best_checkpoint_path}")

# 重新加载base model
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 加载peft adapter
model = PeftModel.from_pretrained(base_model, best_checkpoint_path)
model = model.to("cuda")

# Evaluate！
print("----------------Evaluating!!!---------------")
evaluate(model, tokenizer, valid_dataset,debug=False)