# Snake Game AI using Deep Q Network (DQN)

## Project Overview
This project is an intelligent Snake Game AI that learns to play the classic Snake game automatically using Reinforcement Learning. It implements a Deep Q-Network (DQN) to teach the snake how to find food, avoid walls, and avoid hitting its own tail. The agent learns entirely through trial and error (rewards and penalties) and progressively improves its score over time.

## Reinforcement Learning Explanation
Reinforcement Learning (RL) is an area of machine learning where an agent learns to behave in an environment by performing actions and seeing the results. The agent gets positive rewards for good actions (like eating food) and negative penalties for bad actions (like hitting a wall). Over time, it learns the optimal strategy to maximize its total reward.

## DQN (Deep Q-Network) Explanation
Deep Q-Learning combines standard Q-Learning with Deep Neural Networks. Standard Q-Learning struggles with a large state space, so a neural network is used to approximate the Q-value function. 
- **State**: The input to our neural network (11 parameters: danger directions, current direction, food location).
- **Network**: The hidden layer processes the state, learning patterns (256 neurons).
- **Action**: The output layer determines the best move (Straight, Right, Left).

## Folder Structure
```
snake-ai/
│
├── main.py             # Entry point of the application
├── game.py             # Pygame environment (Snake game logic)
├── agent.py            # AI Agent (Controls the snake, gets states, defines actions)
├── model.py            # PyTorch Neural Network and QTrainer
├── helper.py           # Matplotlib plotting for training curves
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
│
├── models/             # Saved PyTorch models
│   └── model.pth
│
├── graphs/             # Saved performance graphs
│   └── training_plot.png
│
└── assets/             # Images, sounds, etc. (if any)
```

## Installation Steps
1. Make sure you have Python 3.7+ installed.
2. Clone or download this repository.
3. Open a terminal in the project directory.
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Project
To start the AI training, run the `main.py` file:
```bash
python main.py
```
A Pygame window will open showing the snake playing, and a Matplotlib graph will appear updating in real-time with the agent's performance.

## Training Explanation
1. **Get current state**: The agent observes the game board (11 features).
2. **Predict action**: The neural network predicts the best move, or explores randomly.
3. **Perform action**: The snake moves in the chosen direction.
4. **Receive reward**: Eating food (+10), Game Over (-10), otherwise (0).
5. **Train short memory**: The network is trained on this single step.
6. **Store experience**: The state, action, reward, next state, and game over status are saved to Replay Memory.
7. **Train long memory**: After every game, a batch of experiences is randomly sampled and used to train the network further (Experience Replay).

## Evaluation Metrics
- **Reward Score**: Current score of the game.
- **Learning Curve**: Displayed in the real-time Matplotlib graph.
- **Average Score**: The moving average score over all games played.
- **Survival Time**: Correlates with the number of steps before game over.

## Future Improvements
- Implement Convolutional Neural Networks (CNN) to take pixel inputs directly instead of hardcoded state features.
- Add double DQN or Dueling DQN to stabilize training.
- Experiment with hyperparameter tuning (Learning Rate, Gamma, Batch Size).
