import torch
import random
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot
from game import MazeGame, Direction
import numpy as np
from datetime import datetime

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, file_path=None):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet(20, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.is_loaded = False
        if file_path:
            self.load_agent(file_path)

    def get_state(self, game):

        player_pos = game.player_pos
        exit_pos = game.exit_pos

        point_l = (player_pos[0] - 1, player_pos[1])
        point_r = (player_pos[0] + 1, player_pos[1])
        point_u = (player_pos[0], player_pos[1] - 1)
        point_d = (player_pos[0], player_pos[1] + 1)
        
        direction = game.direction
        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN



        state = [
            # Player position
            player_pos[0],
            player_pos[1],

            # Exit position
            exit_pos[0],
            exit_pos[1],

            # Points around player
            game.grid[point_l[1]][point_l[0]],
            game.grid[point_r[1]][point_r[0]],
            game.grid[point_u[1]][point_u[0]],
            game.grid[point_d[1]][point_d[0]],

            #visited before
            (point_l in game.positions_before),
            (point_r in game.positions_before),
            (point_u in game.positions_before),
            (point_d in game.positions_before),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            #END location
            player_pos[0] < exit_pos[0],  # player is left of exit
            player_pos[0] > exit_pos[0],  # player is right of exit
            player_pos[1] < exit_pos[1],  # player is above exit
            player_pos[1] > exit_pos[1],  # player is below exit
        ]
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 1 - self.n_games / 1000
        final_move = [0, 0, 0, 0]
        self.epsilon = 0.1 if self.is_loaded else self.epsilon

        if random.random() < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            print(state)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        

        return final_move
    
    def load_agent(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
        self.model.eval()
        self.is_loaded = True


def train():
    plot_scores = []
    plot_mean_scores = []
    mean_last_100s = []
    total_score = 0
    record = 1000
    agent = Agent()
    game = MazeGame()

    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, steps = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game._reset()
            agent.n_games += 1
            agent.train_long_memory()



            plot_scores.append(steps)
            total_score += steps
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            mean_last_100 = np.mean(plot_scores[-100:])
            mean_last_100s.append(mean_last_100)
            plot(plot_scores, plot_mean_scores, mean_last_100s)

            print(f'Game {agent.n_games} Score: {steps}, Epsilon: {agent.epsilon}, Mean Score: {mean_score}, Mean Last 100: {mean_last_100}')

            if agent.epsilon < -1.5:
                agent.model.save(file_id=game.id, file_name=f"agent{datetime.now().strftime('%Y%m%d%H%M%S')}.pth", best_score=record, mean_score=mean_score, n_games=agent.n_games, epsilon=agent.epsilon)
                plot(plot_scores, plot_mean_scores, './plot11.png')
                break



if __name__ == '__main__':
    train()
