import torch


def SSE(logits, label):
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1
    out = logits - target
    print("IN SSE")
    print(target.requires_grad)
    print(out.requires_grad)
    return out
