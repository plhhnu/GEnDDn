import math
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
class transNet2(nn.Module):
    def __init__(self, in_feature_num,hidden,output):
        super(transNet2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_feature_num, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden // 2, output)
)

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out.squeeze(-1)