import pygame

pygame.init()

# Ustawienia okna
WIDTH, HEIGHT = 600, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Game")

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Rozmiar komórek labiryntu
CELL_SIZE = 20
NUM_CELLS_X = WIDTH // CELL_SIZE
NUM_CELLS_Y = HEIGHT // CELL_SIZE

# Inicjalizacja tablicy labiryntu
grid = []

# Wczytanie labiryntu z pliku
def load_maze():
    global grid
    with open("maze.txt", "r") as f:
        grid = [list(map(int, line.strip().split())) for line in f]
        print(grid)

load_maze()

# Pozycja gracza
player_pos = [0, 0]
start_pos = [0, 0]

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
            elif grid[y][x] == -1:  # Wejście
                pygame.draw.rect(WINDOW, GREEN, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                
            if (x, y) == (player_pos[0], player_pos[1]):  # Gracz
                pygame.draw.circle(WINDOW, BLUE, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2)

    pygame.display.update()

def move_player(dx, dy):
    new_x = player_pos[0] + dx
    new_y = player_pos[1] + dy

    if 0 <= new_x < NUM_CELLS_X and 0 <= new_y < NUM_CELLS_Y and grid[new_y][new_x] != 0:
        player_pos[0] = new_x
        player_pos[1] = new_y

# Pętla główna gry
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                move_player(0, -1)
            elif event.key == pygame.K_DOWN:
                move_player(0, 1)
            elif event.key == pygame.K_LEFT:
                move_player(-1, 0)
            elif event.key == pygame.K_RIGHT:
                move_player(1, 0)

    draw_grid()

pygame.quit()
