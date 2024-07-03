import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt

pygame.init()

# Ustawienia okna
WIDTH, HEIGHT = 600, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DQN Maze Training")

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Rozmiar komórek labiryntu
SIZE = 12
CELL_SIZE = WIDTH // SIZE
NUM_CELLS_X = WIDTH // CELL_SIZE
NUM_CELLS_Y = HEIGHT // CELL_SIZE

# Wczytanie labiryntu z pliku
with open("minimaze.txt", "r") as f:
    grid = [list(map(int, line.strip().split())) for line in f]

def draw_grid(player_pos):
    WINDOW.fill(WHITE)
    for y in range(NUM_CELLS_Y):
        for x in range(NUM_CELLS_X):
            if grid[y][x] == 0:
                pygame.draw.rect(WINDOW, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif grid[y][x] == 1:
                pygame.draw.rect(WINDOW, WHITE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif grid[y][x] == 2:
                pygame.draw.rect(WINDOW, RED, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif grid[y][x] == -1:
                pygame.draw.rect(WINDOW, GREEN, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.draw.rect(WINDOW, BLUE, (player_pos[0] * CELL_SIZE, player_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.display.update()

class MazeEnv:
    def __init__(self, grid, history_length=5):
        self.grid = grid
        self.start_pos = (0, 0)
        self.end_pos = self.find_end_pos()
        self.steps = 0
        self.prev_action = None
        self.history_length = history_length
        self.history = deque(maxlen=history_length)
        self.reset()

    def reset(self):
        self.player_pos = list(self.start_pos)
        self.steps = 0
        self.prev_action = None
        self.history = deque([self.start_pos] * self.history_length, maxlen=self.history_length)
        return self.get_state()

    def find_end_pos(self):
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == 2:
                    return (x, y)
        return None

    def get_state(self):
        # Obliczenie odległości do celu w różnych kierunkach
        x, y = self.player_pos
        goal_x, goal_y = self.end_pos

        dist_straight = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)

        # Odległość po ruchu w lewo
        if self.prev_action == 0:  # left
            new_x = x - 1
            dist_left = np.sqrt((new_x - goal_x) ** 2 + (y - goal_y) ** 2)
        else:
            dist_left = dist_straight

        # Odległość po ruchu w prawo
        if self.prev_action == 1:  # right
            new_x = x + 1
            dist_right = np.sqrt((new_x - goal_x) ** 2 + (y - goal_y) ** 2)
        else:
            dist_right = dist_straight

        # Odległość po ruchu do góry
        if self.prev_action == 2:  # up
            new_y = y - 1
            dist_up = np.sqrt((x - goal_x) ** 2 + (new_y - goal_y) ** 2)
        else:
            dist_up = dist_straight

        # Odległość po ruchu w dół
        if self.prev_action == 3:  # down
            new_y = y + 1
            dist_down = np.sqrt((x - goal_x) ** 2 + (new_y - goal_y) ** 2)
        else:
            dist_down = dist_straight

        return (self.player_pos[0], self.player_pos[1], dist_straight, dist_left, dist_right, dist_up, dist_down)

    def get_available_actions(self):
        actions = []
        x, y = self.player_pos
        if x > 0 and self.grid[y][x - 1] != 0:
            actions.append(0)  # left
        if x < len(self.grid[0]) - 1 and self.grid[y][x + 1] != 0:
            actions.append(1)  # right
        if y > 0 and self.grid[y - 1][x] != 0:
            actions.append(2)  # up
        if y < len(self.grid) - 1 and self.grid[y + 1][x] != 0:
            actions.append(3)  # down
        return actions

    def step(self, action):
        reward = 0
        x, y = self.player_pos
        if action == 0:  # left
            x -= 1
        elif action == 1:  # right
            x += 1
        elif action == 2:  # up
            y -= 1
        elif action == 3:  # down
            y += 1
        
        self.player_pos = [x, y]
        self.history.append(tuple(self.player_pos))

        done = self.player_pos == list(self.end_pos)
        self.steps += 1
        reward -= 1  # kara za każdy krok

        # Kara i nagroda za zbliżanie się do celu
        if self.steps > 1:
            prev_dist = np.sqrt((self.history[-2][0] - self.end_pos[0]) ** 2 + (self.history[-2][1] - self.end_pos[1]) ** 2)
            curr_dist = np.sqrt((self.history[-1][0] - self.end_pos[0]) ** 2 + (self.history[-1][1] - self.end_pos[1]) ** 2)
            if curr_dist < prev_dist:
                reward += 2  # nagroda za zbliżanie się
            else:
                reward -= 14  # kara za oddalanie się

        # Dodatkowe nagrody/ kary
        if done:
            reward += 1000  # nagroda za ukończenie

        self.prev_action = action

        return self.get_state(), reward, done
    
    def get_grid(self):
        return self.grid
    
    def get_player_position(self):
        return self.player_pos

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
    state = env.reset()
    input_dim = len(state)  # Pobierz wymiary wejściowe na podstawie stanu
    output_dim = 4  # liczba akcji
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=10000)

    epsilon = epsilon_start

    rewards = []
    steps = []

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)
        total_reward = 0

        for t in range(1000):  # maksymalna liczba kroków w jednym epizodzie
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            draw_grid(env.get_player_position())

            if random.random() < epsilon:
                action = random.choice(env.get_available_actions())
            else:
                with torch.no_grad():
                    q_values = policy_net(state)
                    action = torch.argmax(q_values).item()
                    if action not in env.get_available_actions():
                        action = random.choice(env.get_available_actions())

            next_state, reward, done = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            total_reward += reward

            memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                break

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards_batch, next_states, dones = zip(*batch)

                states = torch.stack([torch.tensor(s).clone().detach().to(torch.float32) for s in states])
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch).clone().detach().unsqueeze(1).to(torch.float32)
                next_states = torch.stack([torch.tensor(s).clone().detach().to(torch.float32) for s in next_states])
                dones = torch.tensor(dones).clone().detach().unsqueeze(1).to(torch.float32)

                current_q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                expected_q_values = rewards_batch + (gamma * next_q_values * (1 - dones))

                loss = nn.MSELoss()(current_q_values, expected_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        rewards.append(total_reward)
        steps.append(env.steps)

        mean_time_per_episode = (time.time() - current_time) / (episode + 1)
        print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {epsilon}, Estimated time left: {(mean_time_per_episode * (num_episodes - episode)) / 60:.2f} minutes, Steps: {env.steps}")

    # Wykresy na koniec treningu
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    plt.tight_layout()
    plt.show()

    return policy_net

if __name__ == "__main__":
    env = MazeEnv(grid)
    trained_policy = train_dqn(env, num_episodes=600)
    torch.save(trained_policy.state_dict(), "trained_policy3.pth")
    pygame.quit()
