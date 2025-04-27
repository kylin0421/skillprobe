import torch

collector = torch.load("./attention_scores_collector.pt")

print(type(collector))
print(collector.keys())

print(type(collector[0]))
print(len(collector[0]))     #num of samples


print(type(collector[0][0]))         # attention weight for 1 batch
print(collector[6][0].shape)