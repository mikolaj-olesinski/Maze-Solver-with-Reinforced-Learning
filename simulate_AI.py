import pygame
import torch
from enviroment import MazeEnv
from dqn import DQN
import random

pygame.init()

# Ustawienia okna
WIDTH, HEIGHT = 600, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Game with AI")

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Rozmiar komórek labiryntu
SIZE = 12
CELL_SIZE = WIDTH // SIZE
NUM_CELLS_X = WIDTH // CELL_SIZE
NUM_CELLS_Y = HEIGHT // CELL_SIZE

# Wczytanie labiryntu z pliku
with open("minimaze.txt", "r") as f:
    grid = [list(map(int, line.strip().split())) for line in f]

env = MazeEnv(grid)

# Wczytanie wytrenowanej polityki
input_dim = len(env.reset())
output_dim = 4
policy_net = DQN(input_dim, output_dim)
policy_net.load_state_dict(torch.load("trained_policy.pth"))

def draw_grid():
    WINDOW.fill(WHITE)

    for y in range(NUM_CELLS_Y):
        for x in range(NUM_CELLS_X):
            if grid[y][x] == 0:
                pygame.draw.rect(WINDOW, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif grid[y][x] == 1:
                pygame.draw.rect(WINDOW, WHITE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            elif grid[y][x] == 2:  # Wyjście
                pygame.draw.rect(WINDOW, RED, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            if (x, y) == tuple(env.player_pos):  # Gracz
                pygame.draw.circle(WINDOW, BLUE, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2)

    pygame.display.update()

def play_game():
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with torch.no_grad():
            q_values = policy_net(state)
            action = torch.argmax(q_values).item()
            if action not in env.get_available_actions():
                action = random.choice(env.get_available_actions())

        next_state, _, done = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        state = next_state

        draw_grid()

        if done:
            print("AI reached the goal!")
            running = False

    pygame.quit()

if __name__ == "__main__":
    play_game()
