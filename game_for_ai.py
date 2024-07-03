import pygame
import torch
from enviroment import MazeEnv
import random

pygame.init()

# Ustawienia okna
WIDTH, HEIGHT = 600, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))

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

