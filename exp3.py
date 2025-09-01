# Minimal Deep Q-Network (DQN) implementation in PyTorch
# Single-file example for CartPole-v1. Keep modifications minimal.
import random, math, collections
import gymnasium as gym, torch, torch.nn as nn, torch.optim as optim

class DQN(nn.Module):
    def __init__(self, s, a):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(), nn.Linear(128,a))
    def forward(self,x): return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity): self.buf=collections.deque(maxlen=capacity)
    def push(self,*transition): self.buf.append(tuple(transition))
    def sample(self,batch): return random.sample(self.buf,batch)
    def __len__(self): return len(self.buf)

def select_action(net, state, eps, n_actions, device):
    if random.random()<eps: return random.randrange(n_actions)
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(net(s).argmax(dim=1).item())

def optimize(online, target, buffer, optimizer, batch_size, gamma, device):
    if len(buffer)<batch_size: return
    batch = buffer.sample(batch_size)
    s,a,r,ns,done = zip(*batch)
    s = torch.tensor(s, dtype=torch.float32, device=device)
    a = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
    r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
    ns = torch.tensor(ns, dtype=torch.float32, device=device)
    done = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(1)

    q_vals = online(s).gather(1,a)
    with torch.no_grad():
        q_next = target(ns).max(1)[0].unsqueeze(1)
        q_target = r + gamma * q_next * (1 - done)
    loss = nn.functional.mse_loss(q_vals, q_target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

if __name__=='__main__':
    env = gym.make('CartPole-v1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    online = DQN(obs_size, n_actions).to(device)
    target = DQN(obs_size, n_actions).to(device)
    target.load_state_dict(online.state_dict())

    buffer = ReplayBuffer(10000)
    optimizer = optim.Adam(online.parameters(), lr=1e-3)

    episodes = 400
    batch_size = 64
    gamma = 0.99
    eps_start, eps_final, eps_decay = 1.0, 0.01, 500
    target_update = 10

    total_steps = 0
    for ep in range(episodes):
        s, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            eps = eps_final + (eps_start - eps_final) * math.exp(-1. * total_steps / eps_decay)
            a = select_action(online, s, eps, n_actions, device)
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            buffer.push(s, a, r, ns, float(done))
            s = ns
            ep_reward += r
            total_steps += 1
            optimize(online, target, buffer, optimizer, batch_size, gamma, device)
        if ep % target_update == 0:
            target.load_state_dict(online.state_dict())
        if ep%10==0:
            print(f"ep {ep}\treward {ep_reward:.1f}\teps {eps:.3f}")
    env.close()
