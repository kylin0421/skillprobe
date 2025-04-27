from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch

class SkillProbingDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length=128):
        self.samples = self.load_samples(file_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def load_samples(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample['input']
        output_text = sample['output']

        # 拼接 input 和 output
        full_text = input_text + " " + output_text

        # 分别编码 input_text 和 full_text
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = full_encoding["input_ids"].squeeze(0)         # (seq_len)
        attention_mask = full_encoding["attention_mask"].squeeze(0)  # (seq_len)

        labels = input_ids.clone()

        # 把 input部分的位置mask成 -100
        input_length = (input_encoding["attention_mask"] == 1).sum().item()
        labels[:input_length] = -100
        labels[attention_mask == 0] = -100 

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }



def test():
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = SkillProbingDataset("../data/reverse_letter_train.json", tokenizer)

    sample=train_dataset[0]
    print(sample["input_ids"])
    print(sample["labels"])


if __name__ == "__main__":
    test()