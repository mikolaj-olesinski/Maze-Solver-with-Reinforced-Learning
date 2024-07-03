import pygame
from enum import Enum

pygame.init()

# Ustawienia okna
WIDTH, HEIGHT = 600, 600
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))

# Kolory
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CELL_SIZE = 50

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class MazeGame:

    def __init__(self):
        with open("minimaze.txt", "r") as f:
            self.grid = [list(map(int, line.strip().split())) for line in f]

        self.player_pos = self.find_position(-1)
        self.exit_pos = self.find_position(2)
        self.steps = 0
        self.direction = None
        self.positions_before = set()

    def find_position(self, value):
        """Helper function to find the position of a value in the grid."""
        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                if self.grid[y][x] == value:
                    return (x, y)
        return None

    def move(self, action):
        """Move the player in the specified direction."""
        direction = self.action_to_dir(action)
        self.positions_before.add(self.player_pos)

        self.direction = direction
        if direction == Direction.UP:
            new_pos = (self.player_pos[0], self.player_pos[1] - 1)
        elif direction == Direction.DOWN:
            new_pos = (self.player_pos[0], self.player_pos[1] + 1)
        elif direction == Direction.LEFT:
            new_pos = (self.player_pos[0] - 1, self.player_pos[1])
        elif direction == Direction.RIGHT:
            new_pos = (self.player_pos[0] + 1, self.player_pos[1])
        else:
            new_pos = self.player_pos
        self.player_pos = new_pos

    def draw_grid(self):
        """Draw the maze grid on the pygame window."""
        WINDOW.fill(WHITE)

        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                cell_color = WHITE
                if self.grid[y][x] == 0:
                    cell_color = BLACK
                elif self.grid[y][x] == 2:
                    cell_color = RED
                elif self.grid[y][x] == -1:
                    cell_color = GREEN
        

                pygame.draw.rect(WINDOW, cell_color,
                                 (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                
                if (x, y) == self.player_pos:
                    pygame.draw.rect(WINDOW, BLUE,
                                        (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def reset(self):
        """Reset the game to its initial state."""
        self.player_pos = self.find_position(-1)
        self.steps = 0
        self.direction = None
        self.positions_before = set()

    def action_to_dir(self, action):
        """Convert the action to a direction."""

        if action == [1, 0, 0, 0]:
            return Direction.UP
        elif action == [0, 1, 0, 0]:
            return Direction.DOWN
        elif action == [0, 0, 1, 0]:
            return Direction.LEFT
        elif action == [0, 0, 0, 1]:
            return Direction.RIGHT

    def is_collision(self, pos, action=None):
        """Check if a position is a collision (i.e. a wall) if yes return to previous position."""
        x, y = pos
        direction = self.action_to_dir(action)
        if x < 0 or x >= len(self.grid[0]) or y < 0 or y >= len(self.grid) or self.grid[y][x] == 0:
            if direction == Direction.UP:
                self.player_pos = (x, y + 1)
            elif direction == Direction.DOWN:
                self.player_pos = (x, y - 1)
            elif direction == Direction.LEFT:
                self.player_pos = (x + 1, y)
            elif direction == Direction.RIGHT:
                self.player_pos = (x - 1, y)

            return True
        return False

    def distance_to_exit(self):
        """Calculate the Manhattan distance to the exit."""
        return abs(self.player_pos[0] - self.exit_pos[0]) + abs(self.player_pos[1] - self.exit_pos[1])

    def play_step(self, action):
        """Execute one step of the game based on the agent's action."""
        reward = 0
        done = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the player
        self.move(action)

        # Check for collisions
        self.is_collision(self.player_pos, action)
        self.draw_grid() # draw to show collision
        
        if self.steps >= 999:
            done = True
            reward += -1000

        # miuns for going back or for collision
        if self.player_pos in self.positions_before:
            reward += -20
        else:
            reward += 15


        if self.player_pos == self.exit_pos:
            reward += 3000
            done = True


        reward += -5
        self.steps += 1

        # Draw the grid
        self.draw_grid()
        pygame.display.flip()

        return reward, done, self.steps
    
if __name__ == "__main__":
    game = MazeGame()
    game.draw_grid()
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.play_step([1, 0, 0, 0])
                elif event.key == pygame.K_DOWN:
                    game.play_step([0, 1, 0, 0])
                elif event.key == pygame.K_LEFT:
                    game.play_step([0, 0, 1, 0])
                elif event.key == pygame.K_RIGHT:
                    game.play_step([0, 0, 0, 1])
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    quit()
