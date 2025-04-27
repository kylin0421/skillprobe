from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from utils.dataloader import SkillProbingDataset


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

valid_dataset = SkillProbingDataset("./data/reverse_number_valid.json", tokenizer)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)



#########
attention_scores_collector = {}

def save_attention_scores(module, input, output):
    """
    module: 当前attention模块
    input: (query, key, value)
    output: (attn_output, attn_weights) 
    """
    attn_weights = output[1]  # output[1] 是 attention scores
    layer_idx = module.layer_idx
    if layer_idx not in attention_scores_collector:
        attention_scores_collector[layer_idx] = []
    attention_scores_collector[layer_idx].append(attn_weights.detach().cpu())

# 2. 给每一层attention注册hook
hooks = []

for i, layer in enumerate(tqdm(model.model.layers)):  
    attn = layer.self_attn  # TinyLlama和Llama系列通常叫self_attn
    attn.layer_idx = i      # 给每个self-attn模块打上自己是哪一层的标记
    h = attn.register_forward_hook(save_attention_scores)
    hooks.append(h)

print(f"Registered {len(hooks)} attention hooks.")



model.eval()
attention_scores_collector.clear()
with torch.no_grad():
    for batch in tqdm(valid_loader):
        inputs = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**inputs,output_attentions=True)  # 不需要拿输出，只是为了触发forward
        


torch.save(attention_scores_collector, "./number_attention_scores_collector.pt")
print("Attention score collected!")
