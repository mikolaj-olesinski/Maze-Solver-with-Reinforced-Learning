import pygame
import random
from database import add_file

'''
1 - path
0 - wall
-1 - entrance
2 - exit
'''

pygame.init()

WIDTH, HEIGHT = 600, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DFS Maze Generator with Solution Path")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

SIZE = 20
CELL_SIZE = WIDTH // SIZE
NUM_CELLS_X = WIDTH // CELL_SIZE
NUM_CELLS_Y = HEIGHT // CELL_SIZE

grid = [[0] * NUM_CELLS_X for _ in range(NUM_CELLS_Y)]

entrance = (0, 0)

stack = []

def generate_maze():
    current_cell = (0, 0)
    grid[current_cell[1]][current_cell[0]] = -1
    stack.append(current_cell)

    while stack:
        pygame.time.delay(20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        neighbors = []
        x, y = current_cell
        if x > 1 and grid[y][x - 2] == 0: 
            neighbors.append((x - 2, y))
        if x < NUM_CELLS_X - 2 and grid[y][x + 2] == 0: 
            neighbors.append((x + 2, y))
        if y > 1 and grid[y - 2][x] == 0: 
            neighbors.append((x, y - 2))
        if y < NUM_CELLS_Y - 2 and grid[y + 2][x] == 0:
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

    exit_found = False
    for x in range(NUM_CELLS_X - 1, -1, -1): 
        for y in range(NUM_CELLS_Y - 1, -1, -1): 
            if grid[y][x] == 1: 
                grid[y][x] = 2  
                print(f"Znaleziono wyjście na pozycji ({x}, {y})")
                end_x, end_y = x, y
                exit_found = True
                break
        if exit_found:
            break


    draw_grid()
    return grid, end_x, end_y

def draw_grid():
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

    pygame.display.update()


if __name__ == "__main__":

    grid, end_x, end_y = generate_maze()
    grid_to_str = ""
    for row in grid:
        for cell in row:
            grid_to_str += str(cell) + " "
        grid_to_str += "\n"

    add_file("maze", grid_to_str, NUM_CELLS_X, NUM_CELLS_Y, entrance[0], entrance[1], end_x, end_y)


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()
