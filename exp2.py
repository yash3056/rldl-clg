import numpy as np
import gymnasium as gym
from collections import defaultdict

class QLearningAgent:
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.episode_rewards = []
        self.losses = []
        
    def discretize_state(self, state):
        bins = [np.linspace(-2.4, 2.4, 10), np.linspace(-3.0, 3.0, 10), 
                np.linspace(-0.5, 0.5, 10), np.linspace(-2.0, 2.0, 10)]
        return tuple(np.digitize(state[i], bins[i]) for i in range(4))
    
    def get_action(self, state, training=True):
        discrete_state = self.discretize_state(state)
        if training and np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        return np.argmax(self.q_table[discrete_state])
    
    def update_q_table(self, state, action, reward, next_state, done):
        ds, dns = self.discretize_state(state), self.discretize_state(next_state)
        current_q = self.q_table[ds][action]
        target_q = reward if done else reward + self.gamma * np.max(self.q_table[dns])
        td_error = abs(target_q - current_q)
        self.losses.append(td_error)
        self.q_table[ds][action] += self.lr * (target_q - current_q)
        return td_error
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent(episodes=50000, max_steps=500):
    env = gym.make('CartPole-v1')
    agent = QLearningAgent(n_actions=env.action_space.n)
    GOAL_REWARD = 195
    solved = False
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            td_error = agent.update_q_table(state, action, reward, next_state, done)
            episode_loss += td_error
            episode_reward += reward
            state = next_state
            steps += 1
            if done:
                break
        
        agent.decay_epsilon()
        agent.episode_rewards.append(episode_reward)
        
        if len(agent.episode_rewards) >= 100:
            recent_avg = np.mean(agent.episode_rewards[-100:])
            if recent_avg >= GOAL_REWARD and not solved:
                solved = True
                print(f"GOAL ACHIEVED at episode {episode + 1}! Avg reward: {recent_avg:.2f}")
        
        if (episode + 1) % 5000 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_loss = episode_loss / steps if steps > 0 else 0
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Loss = {avg_loss:.4f}, Epsilon = {agent.epsilon:.3f}")
    
    env.close()
    return agent

def main():
    print("Q-Learning Implementation for CartPole")
    agent = train_agent(episodes=50000)
    
    avg_reward = np.mean(agent.episode_rewards[-100:])
    avg_loss = np.mean(agent.losses[-1000:]) if agent.losses else 0
    print(f"Final average reward: {avg_reward:.2f}")
    print(f"Final average loss: {avg_loss:.4f}")
    print(f"Q-table states explored: {len(agent.q_table)}")
    
    if avg_reward >= 195:
        print("SUCCESS: Agent learned to balance the pole!")
    else:
        print("Agent didn't fully converge. Try training longer.")
    
    return agent

if __name__ == "__main__":
    trained_agent = main()