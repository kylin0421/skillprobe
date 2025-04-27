from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
from tqdm import tqdm
from utils.dataloader import SkillProbingDataset
import torch.distributed as dist

def evaluate(model, tokenizer, valid_dataset, max_new_tokens=20,debug=False):
    
    if dist.is_initialized() and dist.get_rank() != 0:
        # 非rank0的GPU直接跳过evaluation
        return

    model.eval()
    correct = 0
    total = 0

    for i, sample in enumerate(tqdm(valid_dataset)):
        input_ids = sample["input_ids"]    # (注意，这里不unsqueeze)
        labels = sample["labels"]
        attention_mask = sample["attention_mask"]





        prompt_len = (labels != -100).nonzero()[0].item()
        output_label_len = (labels != -100).sum().item()


        # 取prompt部分的input_ids和attention_mask，并unsqueeze
        prompt_input_ids = input_ids[:prompt_len].unsqueeze(0).to(model.device)
        prompt_attention_mask = attention_mask[:prompt_len].unsqueeze(0).to(model.device)

        prompt_text = tokenizer.decode(prompt_input_ids[0], skip_special_tokens=True)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=output_label_len,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if generated_text.startswith(prompt_text):
            generated_answer = generated_text[len(prompt_text):].strip()
        else:
            generated_answer = generated_text.strip()

        expected_output = tokenizer.decode(
            labels[labels != -100],
            skip_special_tokens=True
        ).strip()


        if debug:
            if i % 30 == 0:
                print(f"Sample {i}:")
                print(f"  Prompt: {prompt_text}")
                print(f"  Generated: {generated_answer}")
                print(f"  Expected: {expected_output}")
                return

        if generated_answer == expected_output:
            correct += 1
        total += 1



    acc = correct / total
    print(f"\nFinal Accuracy: {correct}/{total} = {acc:.4f}")
    return acc



if __name__=="__main__":
    model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载peft adapter
    use_ckpt=True
    if use_ckpt:
        best_checkpoint_path="./out/prefix_reverse_letter/checkpoint-500"
        model = PeftModel.from_pretrained(base_model, best_checkpoint_path)
        model = model.to("cuda")

    else:
        model=base_model

    valid_dataset = SkillProbingDataset("./data/reverse_combined_valid.json", tokenizer)

    evaluate(model,tokenizer,valid_dataset)