import torch


def SSE(logits, label):
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1
    print("Single errors")
    print(0.5 * ((logits - target) ** 2))
    out = 0.5 * ((logits - target) ** 2).sum()
    return out
