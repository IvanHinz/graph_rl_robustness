import torch.nn as nn

# DQN neural network
class DQN(nn.Module):
    def __init__(self, observation_shape, hidden_size=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(observation_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, observation_shape)
        )

    def forward(self, x):
        return self.fc(x)

# actor critic
# initially the same layers
# then we get 2 outputs - policy with shape of action space (here the same as observation_space shape)
# and value of critic
class ActorCritic(nn.Module):
    def __init__(self, observation_shape, hidden_size=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(observation_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.policy = nn.Linear(hidden_size, observation_shape)
        self.value  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc(x)
        logits = self.policy(x)
        value  = self.value(x).squeeze(-1)
        return logits, value

