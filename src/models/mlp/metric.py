import torch

class TopKAccuracy:
    def __init__(self, k=1):
        self.k = k

    def compute(self, output, target):
        
        correct = 0
        correct = (output == target).sum().item()
        return correct / len(target)
    
    def __str__(self):
        return f"top{self.k}"