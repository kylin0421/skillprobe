import random
import json

# 设置随机种子，保证复现
random.seed(42)

# 生成 Copy 任务
def generate_copy_task(n_samples=500, seq_len=10):
    samples = []
    for _ in range(n_samples):
        input_seq = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=seq_len))
        samples.append({
            "input": f"Copy the following sequence: {input_seq}",
            "output": input_seq
        })
    return samples

# 生成 Reverse 任务
def generate_reverse_letter_task(n_samples=500, seq_len=10):
    samples = []
    seen_inputs = set()
    for _ in range(n_samples):
        input_seq = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=seq_len))
        if input_seq in seen_inputs:
            continue  # 已经生成过了，跳过
        seen_inputs.add(input_seq)
        reversed_seq = input_seq[::-1]
        samples.append({
            "input": f"Reverse the following sequence: {input_seq}",
            "output": reversed_seq
        })
    return samples

def generate_reverse_number_task(n_samples=500, seq_len=10):
    samples = []
    seen_inputs = set()
    for _ in range(n_samples):
        input_seq = ''.join(random.choices('0123456789', k=seq_len))
        if input_seq in seen_inputs:
            continue  # 已经生成过了，跳过
        seen_inputs.add(input_seq)
        reversed_seq = input_seq[::-1]
        samples.append({
            "input": f"Reverse the following sequence: {input_seq}",
            "output": reversed_seq
        })
    return samples

def generate_reverse_combined_task(n_samples=500, seq_len=10):
    samples = []
    seen_inputs=set()
    for _ in range(n_samples):
        input_seq = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=seq_len))
        if input_seq in seen_inputs:
            continue  # 已经生成过了，跳过
        seen_inputs.add(input_seq)
        reversed_seq = input_seq[::-1]
        samples.append({
            "input": f"Reverse the following sequence: {input_seq}",
            "output": reversed_seq
        })
    return samples



# 生成 Sort 任务
def generate_sort_task(n_samples=500, num_range=(0, 9), seq_len=10):
    samples = []
    for _ in range(n_samples):
        numbers = [str(random.randint(*num_range)) for _ in range(seq_len)]
        input_seq = ' '.join(numbers)
        sorted_seq = ' '.join(sorted(numbers, key=int))
        samples.append({
            "input": f"Sort the following numbers in ascending order: {input_seq}",
            "output": sorted_seq
        })
    return samples

# 生成简单加法任务
def generate_add_task(n_samples=500, num_range=(0, 200)):
    samples = []
    for _ in range(n_samples):
        a, b = random.randint(*num_range), random.randint(*num_range)
        samples.append({
            "input": f"What is {a} + {b}?",
            "output": str(a + b)
        })
    return samples


num_samples=10000
seq_len=5
basic_skill_tasks = {
    "copy": generate_copy_task(n_samples=num_samples,seq_len=seq_len),
    "reverse_letter": generate_reverse_letter_task(n_samples=num_samples,seq_len=seq_len),
    "reverse_number": generate_reverse_number_task(n_samples=num_samples,seq_len=seq_len),
    "reverse_combined": generate_reverse_combined_task(n_samples=num_samples,seq_len=seq_len),
    "sort": generate_sort_task(n_samples=num_samples,seq_len=seq_len),
    "add": generate_add_task(n_samples=num_samples)
}


# 整合生成所有 basic tasks
# 保存train/valid
for task_name, samples in basic_skill_tasks.items():
    random.shuffle(samples)  # 打乱样本顺序
    
    split_ratio = 0.9
    train_size = int(len(samples) * split_ratio)
    
    train_samples = samples[:train_size]
    valid_samples = samples[train_size:]
    
    # 保存 train
    with open(f"{task_name}_train.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    
    # 保存 valid
    with open(f"{task_name}_valid.json", "w") as f:
        json.dump(valid_samples, f, indent=2)

print("All tasks saved into train/valid splits!")


