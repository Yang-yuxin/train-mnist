import torch
import torch.nn.functional as F

class softmax_model(torch.nn.Module):
    def __init__(self, n_fc):
        super(softmax_model,self).__init__()
        self.fc = torch.nn.Linear(n_fc[0], n_fc[1])
        self._in = n_fc[0]
        self._out = n_fc[1]

    def forward(self, x):
        x = x.view((-1, self._in))
        x = self.fc(x)
        return x



