import torch
from torch import nn

class SoftmaxClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        return logits

#Logistic Regression - All vs One
class LogisticRegression(nn.Module):
    def __init__(self, in_features):
        super(LogisticRegression, self).__init__()
        
        self.linear = nn.Linear(in_features, 1)
        #self.logistic = nn.Sigmoid()

    def forward(self, x):
        #logits = self.linear(x)
        #return self.logistic(logits)

        return self.linear(x) #SOL A : SBILANCIAMENTO DELLE CLASSI