import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import gym
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from ncps.datasets.torch import AtariCloningDataset
from ncps.torch import CfC, CfCCell
from ncps.wirings.wirings import Wiring
import torch
from torch import nn
from typing import Optional, Union
import ncps
from ncps.torch.lstm import LSTMCell 

# ConvBlock definition
class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 64, 5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 5, padding=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 5, padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.mean((-1, -2))  # Global average pooling
        return x

class ConvCfC(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv_block = ConvBlock()
        self.rnn = CfC(256, 64, batch_first=True, proj_size=n_actions)

    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # Merge time and batch dimension into a single one (because the Conv layers require this)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)  # apply conv block to merged data
        # Separate time and batch dimension again
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.rnn(x, hx)  # hx is the hidden state of the RNN
        return x, hx

def run_closed_loop(model, env, num_episodes=None):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    device = next(model.parameters()).device
    hx = None  # Hidden state of the RNN
    returns = []
    total_reward = 0
    with torch.no_grad():
        while True:
            print(f"Original obs shape: {obs.shape}")  # 打印原始 obs 形狀
            obs = np.asarray(obs)
            if len(obs.shape) == 3:  # 確保 obs 是 3 維
                obs = np.transpose(obs, [2, 0, 1]).astype(np.float32) / 255.0
            else:
                raise ValueError(f"Unexpected obs shape: {obs.shape}")
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)
            pred, hx = model(obs, hx)
            action = pred.squeeze(0).squeeze(0).argmax().item()
            
            result = env.step(action)
            if len(result) == 4:
                obs, r, done, info = result
            else:
                obs, r, done, info, _ = result
            
            if isinstance(obs, tuple):
                obs = obs[0]
            total_reward += r
            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                hx = None  # Reset hidden state of the RNN
                returns.append(total_reward)
                total_reward = 0
                if num_episodes is not None:
                    num_episodes = num_episodes - 1
                    if num_episodes == 0:
                        return returns

def eval(model, valloader):
    losses, accs = [], []
    model.eval()
    device = next(model.parameters()).device  # get device the model is located on
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs = inputs.to(device)  # move data to same device as the model
            labels = labels.to(device)

            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
            labels = labels.view(-1, *labels.shape[2:])  # flatten
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(-1) == labels).float().mean()
            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)

# Initialize environment
env = gym.make("ALE/Breakout-v5")
criterion = nn.CrossEntropyLoss()
env = wrap_deepmind(env)
device = torch.device("cpu")
loaded_model = ConvCfC(n_actions=env.action_space.n).to(device)
loaded_model.load_state_dict(torch.load('cfc_model_wiring.pt', map_location=torch.device('cpu')))
loaded_model.eval()
print("Model loaded from cfc_model.pt")

# 確保模型處於評估模式
# Visualize Atari game and play endlessly
env = gym.make("ALE/Breakout-v5", render_mode="human")
env = wrap_deepmind(env)
run_closed_loop(loaded_model, env)
# 再次運行閉環測試
returns = run_closed_loop(loaded_model, env, num_episodes=10)
print(f"Mean return {np.mean(returns)} (n={len(returns)})")

