# 复现方法

## 环境准备

1. 创建并激活一个新的虚拟环境：
    ```sh
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

2. 安装所需的Python库：
    ```sh
    pip install -r requirements.txt
    ```

## 模型训练和评估

1. 准备数据集和环境：
    ```python
    import gym
    from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
    from ncps.datasets.torch import AtariCloningDataset
    from torch.utils.data import DataLoader

    # 初始化环境
    env = gym.make("ALE/Breakout-v5")
    env = wrap_deepmind(env)

    # 准备数据集
    train_ds = AtariCloningDataset("breakout", split="train")
    val_ds = AtariCloningDataset("breakout", split="val")
    trainloader = DataLoader(train_ds, batch_size=32, num_workers=4, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=32, num_workers=4)
    ```

2. 定义和初始化模型：
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from ncps.torch import CfC, CfCCell
    from ncps.wirings import WiringRevised

    # 定义卷积神经网络模块
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
            x = x.mean((-1, -2))  # 全局平均池化
            return x

    # 定义 ConvCfC 模型
    class ConvCfC(nn.Module):
        def __init__(self, n_actions):
            super().__init__()
            self.conv_block = ConvBlock()
            self.rnn = CfC(256, 64, batch_first=True, proj_size=n_actions)

        def forward(self, x, hx=None):
            batch_size = x.size(0)
            seq_len = x.size(1)
            # 合并时间和批次维度
            x = x.view(batch_size * seq_len, *x.shape[2:])
            x = self.conv_block(x)  # 应用卷积块
            # 恢复时间和批次维度
            x = x.view(batch_size, seq_len, *x.shape[1:])
            x, hx = self.rnn(x, hx)  # hx 是 RNN 的隐藏状态
            return x, hx

    # 初始化设备和模型
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = ConvCfC(n_actions=env.action_space.n).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    ```

3. 训练模型并保存日志：
    ```python
    from tqdm import tqdm

    def train_one_epoch(model, criterion, optimizer, trainloader):
        running_loss = 0.0
        pbar = tqdm(total=len(trainloader))
        model.train()
        device = next(model.parameters()).device  # 获取模型所在设备
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)  # 将数据移至与模型相同的设备
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, hx = model(inputs)
            labels = labels.view(-1, *labels.shape[2:])  # 展平
            outputs = outputs.reshape(-1, *outputs.shape[2:])  # 展平
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_description(f"loss={running_loss / (i + 1):0.4g}")
            pbar.update(1)
        pbar.close()

    def eval(model, valloader):
        losses, accs = [], []
        model.eval()
        device = next(model.parameters()).device  # 获取模型所在设备
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(device)  # 将数据移至与模型相同的设备
                labels = labels.to(device)

                outputs, _ = model(inputs)
                outputs = outputs.reshape(-1, *outputs.shape[2:])  # 展平
                labels = labels.view(-1, *labels.shape[2:])  # 展平
                loss = criterion(outputs, labels)
                acc = (outputs.argmax(-1) == labels).float().mean()
                losses.append(loss.item())
                accs.append(acc.item())
        return np.mean(losses), np.mean(accs)

    log_file = open("training_log_wiring.txt", "a")
    for epoch in range(20):
        train_one_epoch(model, criterion, optimizer, trainloader)
        val_loss, val_acc = eval(model, valloader)
        log_message = f"Epoch {epoch+1}, val_loss={val_loss:0.4g}, val_acc={100*val_acc:0.2f}%\n"
        print(log_message)
        log_file.write(log_message)
    log_file.close()
    ```

4. 运行闭环测试：
    ```python
    def run_closed_loop(model, env, num_episodes=None):
        obs = env.reset()
        device = next(model.parameters()).device
        hx = None  # RNN的隐藏状态
        returns = []
        total_reward = 0
        with torch.no_grad():
            while True:
                obs = np.transpose(obs, [2, 0, 1]).astype(np.float32) / 255.0
                obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)
                pred, hx = model(obs, hx)
                action = pred.squeeze(0).squeeze(0).argmax().item()
                obs, r, done, _ = env.step(action)
                total_reward += r
                if done:
                    obs = env.reset()
                    hx = None  # 重置 RNN 的隐藏状态
                    returns.append(total_reward)
                    total_reward = 0
                    if num_episodes is not None:
                        num_episodes = num_episodes - 1
                        if num_episodes == 0:
                            return returns

    loaded_model = ConvCfC(n_actions=env.action_space.n).to(device)
    loaded_model.load_state_dict(torch.load('model_WiringRevised.pt'))
    loaded_model.eval()
    returns = run_closed_loop(loaded_model, env, num_episodes=10)
    print(f"Mean return {np.mean(returns)} (n={len(returns)})")
    ```

## 注意事项
- 确保CUDA设备可用，或者修改代码以使用CPU。
- 请根据实际情况调整数据集路径和文件名。
