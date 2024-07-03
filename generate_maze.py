import pygame
import random

pygame.init()

# Ustawienia okna
WIDTH, HEIGHT = 600, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DFS Maze Generator with Solution Path")

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Rozmiar komórek labiryntu
SIZE = 12
CELL_SIZE = WIDTH // SIZE
NUM_CELLS_X = WIDTH // CELL_SIZE
NUM_CELLS_Y = HEIGHT // CELL_SIZE

# Inicjalizacja tablicy labiryntu
grid = [[0] * NUM_CELLS_X for _ in range(NUM_CELLS_Y)]

# Ustalenie wejścia
entrance = (0, 0)

# Stos do śledzenia odwiedzonych komórek
stack = []

def generate_maze():
    # Ustawienie startowej komórki
    current_cell = (0, 0)
    grid[current_cell[1]][current_cell[0]] = -1
    stack.append(current_cell)

    while stack:
        pygame.time.delay(1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        neighbors = []
        # Sprawdzenie sąsiadów
        x, y = current_cell
        if x > 1 and grid[y][x - 2] == 0:  # Left
            neighbors.append((x - 2, y))
        if x < NUM_CELLS_X - 2 and grid[y][x + 2] == 0:  # Right
            neighbors.append((x + 2, y))
        if y > 1 and grid[y - 2][x] == 0:  # Up
            neighbors.append((x, y - 2))
        if y < NUM_CELLS_Y - 2 and grid[y + 2][x] == 0:  # Down
            neighbors.append((x, y + 2))

        if neighbors:
            next_cell = random.choice(neighbors)
            stack.append(next_cell)
            grid[next_cell[1]][next_cell[0]] = 1
            grid[(current_cell[1] + next_cell[1]) // 2][(current_cell[0] + next_cell[0]) // 2] = 1
            current_cell = next_cell
        else:
            current_cell = stack.pop()

        draw_grid()

    # Znalezienie najbliższego białego pola do prawego dolnego rogu
    exit_found = False
    for x in range(NUM_CELLS_X - 1, -1, -1):  # Od prawej do lewej
        for y in range(NUM_CELLS_Y - 1, -1, -1):  # Od dolu do góry
            if grid[y][x] == 1:  # Białe pole
                grid[y][x] = 2  # Ustawienie wyjścia
                print(f"Znaleziono wyjście na pozycji ({x}, {y})")
                exit_found = True
                break
        if exit_found:
            break


    draw_grid()
    return grid

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
            elif grid[y][x] == -1:
                pygame.draw.rect(WINDOW, GREEN, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.update()


if __name__ == "__main__":

    # Uruchomienie generowania labiryntu
    grid = generate_maze()
    for row in grid:
        for cell in row:
            print(cell, end=" ")
        print("\n")


    # Pętla główna programu
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
