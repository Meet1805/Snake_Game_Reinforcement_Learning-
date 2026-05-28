import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer, save_checkpoint, load_checkpoint
from helper import plot
import os
import json
from datetime import datetime

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

def save_statistics(n_games, record, total_score, plot_scores, plot_mean_scores, epsilon):
    """Saves training progress statistics to JSON to persist graph and game counts."""
    folder = './models'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    data = {
        'n_games': n_games,
        'record': record,
        'total_score': total_score,
        'plot_scores': plot_scores,
        'plot_mean_scores': plot_mean_scores,
        'epsilon': epsilon,
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Write to temp file then rename for safe saving (prevent corruption)
    temp_file = os.path.join(folder, 'training_stats.tmp')
    final_file = os.path.join(folder, 'training_stats.json')
    with open(temp_file, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(temp_file, final_file)

def load_statistics():
    """Loads training statistics from JSON. Returns None if not found or corrupted."""
    file_path = './models/training_stats.json'
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Warning: Could not load training_stats.json: {e}")
            return None
    return None

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        # We handle loading in train() to orchestrate stats and model loading simultaneously


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    agent = Agent()
    game = SnakeGameAI()
    
    # Initialize defaults
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    # Attempt to load previous training session
    print("=========================================")
    print("Initializing Professional RL System...")
    stats = load_statistics()
    loaded_model = load_checkpoint(agent.model, agent.trainer.optimizer)
    
    if loaded_model and stats:
        print("-> Found existing checkpoint and stats! Resuming training...")
        agent.n_games = stats.get('n_games', 0)
        record = stats.get('record', 0)
        total_score = stats.get('total_score', 0)
        plot_scores = stats.get('plot_scores', [])
        plot_mean_scores = stats.get('plot_mean_scores', [])
        agent.epsilon = stats.get('epsilon', 0)
        print(f"-> Resuming from Game {agent.n_games + 1} | Current Record: {record}")
    else:
        print("-> No valid checkpoint found. Starting fresh training from Game 1.")
    print("=========================================")
    
    # Ensure graphs directory exists
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
        
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            is_best = False
            if score > record:
                record = score
                is_best = True
                
            # Show Professional Output
            if agent.n_games == 1:
                print(f"Initial game completed. Score: {score}")
            else:
                current_mean = total_score / agent.n_games if agent.n_games > 0 else 0
                improvement_str = f" | Best: {record} | Mean: {current_mean:.2f}"
                print(f'Game {agent.n_games} | Score: {score}{improvement_str}')

            # Periodically or every time on done:
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            # Auto-save Checkpoint & Stats
            save_checkpoint(agent.model, agent.trainer.optimizer, is_best=is_best)
            save_statistics(
                n_games=agent.n_games, 
                record=record, 
                total_score=total_score, 
                plot_scores=plot_scores, 
                plot_mean_scores=plot_mean_scores,
                epsilon=agent.epsilon
            )

            # Extra: Timestamped backup model every 50 games
            if agent.n_games % 50 == 0:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_checkpoint(agent.model, agent.trainer.optimizer, is_best=False, filename=f'checkpoint_{agent.n_games}_{ts}.pth')

            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
