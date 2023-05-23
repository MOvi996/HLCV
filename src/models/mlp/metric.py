import torch

class TopKAccuracy:
    def __init__(self, k=1):
        self.k = k

    def compute(self, output, target):
        
        correct = 0
        _, predicted = output.topk(self.k, dim=1)
        predicted = predicted.t()
        target_ = target.view(1,-1)
        
        target_ = target_.expand_as(predicted)
        correct = (predicted == target_)[:self.k].reshape(-1).float().sum(0)

        return correct / len(target)
    
    def __str__(self):
        return f"top{self.k}"