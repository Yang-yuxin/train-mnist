import torch
import torch.nn.functional as F

class fc_model(torch.nn.Module):
    def __init__(self, n_fc):
        super(fc_model,self).__init__()
        self.fc1 = torch.nn.Linear(n_fc[0], n_fc[1])
        self.fc2 = torch.nn.Linear(n_fc[1], n_fc[2])
        self.fc3 = torch.nn.Linear(n_fc[2], n_fc[3])
        self.n_fc = n_fc

    def forward(self, x):
        x = x.view((-1, self.n_fc[0]))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        # x = self.fc5(x)
        # x = F.softmax(x, dim=1)
        return x



