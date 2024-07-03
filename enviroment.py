class MazeEnv:
    def __init__(self, grid):
        self.grid = grid
        self.start_pos = (0, 0)
        self.end_pos = self.find_end_pos()
        self.steps = 0
        self.prev_action = None
        self.reset()

    def reset(self):
        self.player_pos = list(self.start_pos)
        self.steps = 0
        self.prev_action = None
        return self.get_state()

    def find_end_pos(self):
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == 2:
                    return (x, y)
        return None

    def get_state(self):
        return tuple(self.player_pos)

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
        x, y = self.player_pos
        if action == 0:  # left
            x -= 1
        elif action == 1:  # right
            x += 1
        elif action == 2:  # up
            y -= 1
        elif action == 3:  # down
            y += 1
        
        if self.prev_action:
            if action == 0 and self.prev_action == 1 or action == 1 and self.prev_action == 0 or action == 2 and self.prev_action == 3 or action == 3 and self.prev_action == 2:
                reward = -8

        self.prev_action = action
        self.player_pos = [x, y]

        done = self.player_pos == list(self.end_pos)
        self.steps += 1
        reward = -1

        if done:
            reward = 1500 - self.steps 

        return self.get_state(), reward, done

    def get_grid(self):
        return self.grid

    def get_player_position(self):
        return self.player_pos
