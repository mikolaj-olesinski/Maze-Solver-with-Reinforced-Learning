import numpy as np
from collections import deque

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

        self.prev_action = action

        return self.get_state(), reward, done
