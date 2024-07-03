import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from enviroment import MazeEnv
import time

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_dqn(env, num_episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, lr=0.001, batch_size=64, target_update=10):
    print("Training DQN...")
    current_time = time.time()
    input_dim = len(env.reset())
    output_dim = 4  # liczba akcji
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=10000)

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0

        for t in range(1000):  # maksymalna liczba kroków w jednym epizodzie
            if random.random() < epsilon:
                action = random.choice(env.get_available_actions())
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = torch.argmax(q_values).item()
                    if action not in env.get_available_actions():
                        action = random.choice(env.get_available_actions())
                    else:
                        print("Action available")

            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward

            memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                break

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
                next_states = torch.stack(next_states)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                current_q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                expected_q_values = rewards + (gamma * next_q_values * (1 - dones))

                loss = nn.MSELoss()(current_q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        mean_time_per_episode = (time.time() - current_time) / (episode + 1)
        print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon}, Estimated time left: {(mean_time_per_episode * (num_episodes - episode)) / 60:.2f} minutes")

    return policy_net

if __name__ == "__main__":
    # Wczytanie labiryntu z pliku
    with open("minimaze.txt", "r") as f:
        grid = [list(map(int, line.strip().split())) for line in f]

    env = MazeEnv(grid)
    trained_policy = train_dqn(env)
    torch.save(trained_policy.state_dict(), "trained_policy.pth")
