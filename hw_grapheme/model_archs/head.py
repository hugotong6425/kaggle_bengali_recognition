from torch import nn


class Head_1fc(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Head_1fc, self).__init__()
        self.fc = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        return self.fc(x)
