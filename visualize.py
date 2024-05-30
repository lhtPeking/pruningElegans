import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from ncps.wirings.wiringsRevised import WiringRevised
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind

from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from ncps.datasets.torch import AtariCloningDataset
from ncps.torch import CfC, CfCCell
from ncps.wirings.wiringsRevised import WiringRevised
import torch
from torch import nn
from typing import Optional, Union
import ncps
from ncps.torch.lstm import LSTMCell 
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
from ncps.wirings.wiringsRevised import WiringRevised
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

# ConvCfC definition
class ConvCfC(nn.Module):
    def __init__(self, n_actions, wiring):
        super().__init__()
        self.conv_block = ConvBlock()
        self.rnn = CfC(256, wiring, batch_first=True, proj_size=n_actions)

    def forward(self, x, hx=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size * seq_len, *x.shape[2:])
        x = self.conv_block(x)  # apply conv block to merged data
        x = x.view(batch_size, seq_len, *x.shape[1:])
        x, hx = self.rnn(x, hx)  # hx is the hidden state of the RNN
        return x, hx

class WiredCfCCell(nn.Module):
    def __init__(self, input_size, wiring, mode="default"):
        super(WiredCfCCell, self).__init__()
        self.input_size = input_size
        self.wiring = wiring
        self.mode = mode
        self.units = wiring.units

        # 定義其他需要的屬性和層
        self.linear = nn.Linear(input_size, self.units)
        self.output_linear = nn.Linear(self.units, wiring.output_dim if wiring.output_dim else self.units)
        
        # 使用 wiring 的 adjacency_matrix 初始化連接
        self.adjacency_matrix = wiring.adjacency_matrix
        self.sensory_adjacency_matrix = wiring.sensory_adjacency_matrix

    def forward(self, x, hx, ts=1.0):
        h = F.relu(self.linear(x) + torch.matmul(hx, self.adjacency_matrix))
        h = F.relu(h + torch.matmul(x, self.sensory_adjacency_matrix))
        if self.mode == "default":
            h = self.output_linear(h)
        return h, h

class CfC(nn.Module):
    def __init__(
        self,
        input_size: Union[int, ncps.wirings.Wiring],
        units,
        proj_size: Optional[int] = None,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        backbone_dropout: Optional[int] = None,
    ):
        super(CfC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.proj_size = proj_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        if isinstance(units, ncps.wirings.Wiring) or isinstance(units, ncps.wirings.wiringsRevised.WiringRevised):
            self.wired_mode = True
            if backbone_units is not None:
                raise ValueError(f"Cannot use backbone_units in wired mode")
            if backbone_layers is not None:
                raise ValueError(f"Cannot use backbone_layers in wired mode")
            if backbone_dropout is not None:
                raise ValueError(f"Cannot use backbone_dropout in wired mode")
            self.wiring = units
            self.state_size = self.wiring.units
            self.output_size = self.wiring.output_dim if self.wiring.output_dim is not None else self.state_size
            self.rnn_cell = WiredCfCCell(
                input_size,
                self.wiring_or_units,
                mode,
            )
        else:
            self.wired_false = True
            backbone_units = 128 if backbone_units is None else backbone_units
            backbone_layers = 1 if backbone_layers is None else backbone_layers
            backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout
            self.state_size = units
            self.output_size = self.state_size
            self.rnn_cell = CfCCell(
                input_size,
                self.wiring_or_units,
                mode,
                activation,
                backbone_units,
                backbone_layers,
                backbone_dropout,
            )
        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = LSTMCell(input_size, self.state_size)

        if proj_size is None:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.output_size, self.proj_size)

    def forward(self, input, hx=None, timespans=None):
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = torch.zeros((batch_size, self.state_size), device=device) if self.use_mixed else None
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError(
                    "Running a CfC with mixed_memory=True, requires a tuple (h0,c0) to be passed as state (got torch.Tensor instead)"
                )
            h_state, c_state = hx if self.use_mixed else (hx, None)
            if is_batched:
                if h_state.dim() != 2:
                    msg = f"For batched 2-D input, hx and cx should also be 2-D but got ({h_state.dim()}-D) tensor"
                    raise RuntimeError(msg)
            else:
                if h_state.dim() != 1:
                    msg = f"For unbatched 1-D input, hx and cx should also be 1-D but got ({h_state.dim()}-D) tensor"
                    raise RuntimeError(msg)
                h_state = h_state.unsqueeze(0)
                c_state = c_state.unsqueeze(0) if c_state is not None else None

        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if self.return_sequences:
                output_sequence.append(self.fc(h_out))

        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = self.fc(h_out)
        hx = (h_state, c_state) if self.use_mixed else h_state

        if not is_batched:
            readout = readout.squeeze(batch_dim)
            hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]

        return readout, hx
    

# Initialize environment
env = gym.make("ALE/Breakout-v5")
env = wrap_deepmind(env)
'''
# Prepare dataset
train_ds = AtariCloningDataset("breakout", split="train")
val_ds = AtariCloningDataset("breakout", split="val")
trainloader = DataLoader(train_ds, batch_size=32, num_workers=4, shuffle=True)
valloader = DataLoader(val_ds, batch_size=32, num_workers=4)
'''
# Training and evaluation functions
def train_one_epoch(model, criterion, optimizer, trainloader):
    running_loss = 0.0
    pbar = tqdm(total=len(trainloader))
    model.train()
    device = next(model.parameters()).device  # get device the model is located on
    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)  # move data to same device as the model
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs, hx = model(inputs)
        labels = labels.view(-1, *labels.shape[2:])  # flatten
        outputs = outputs.reshape(-1, *outputs.shape[2:])  # flatten
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        pbar.set_description(f"loss={running_loss / (i + 1):0.4g}")
        pbar.update(1)
    pbar.close()

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
                    
wiring = WiringRevised(units=256)  # Example wiring
criterion = nn.CrossEntropyLoss()
device = torch.device("cpu")
loaded_model = ConvCfC(n_actions=env.action_space.n, wiring=wiring).to(device)
loaded_model.load_state_dict(torch.load('cfc_model.pt', map_location=torch.device('cpu')))
print("Model loaded from cfc_model.pt")
import os

# 創建 requirements.txt 文件
os.system("pip freeze > VisualizationRequirements.txt")

# 確保模型處於評估模式
loaded_model.eval()
# Visualize Atari game and play endlessly
env = gym.make("ALE/Breakout-v5", render_mode="human")
env = wrap_deepmind(env)
run_closed_loop(loaded_model, env)
# 再次運行閉環測試
returns = run_closed_loop(loaded_model, env, num_episodes=10)
print(f"Mean return {np.mean(returns)} (n={len(returns)})")

