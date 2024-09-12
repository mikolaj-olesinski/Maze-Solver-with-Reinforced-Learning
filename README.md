# Maze Solver with Reinforced Learning using PyTorch

## Project description

This project is a Maze Solver AI that uses Reinforcement Learning (specifically Q-learning) to train an agent to navigate a maze and reach the exit efficiently. The agent learns through trial and error, receiving rewards for successful moves and penalties for poor decisions, with the aim of improving its performance over time. The core of the solution uses Deep Q-Learning with a neural network to approximate the Q-values, and leverages PyTorch for model building and training.

## Feautrers

- **Azure SQL**: The system is built on Azure SQL, offering real-time scalability, high data availability, and robust security. With automatic resource scaling, it efficiently handles large datasets and ensures optimal database management.
Reinforcement Learning: The system employs reinforcement learning techniques, allowing the algorithm to learn and adjust dynamically based on performance results. This leads to continuous optimization over time.
- **Deep Q-Learning Agent**: The agent utilizes a neural network to learn the optimal actions within the maze using reinforcement learning principles, specifically Deep Q-Learning.
- **Deep Q-Learning Agent**: The agent uses a neural network to learn the optimal actions in the maze by reinforcement learning
- **Experience Replay**: It stores past experiences and samples them during training for improved performance.
- **Dynamic Maze Generation**: A maze is randomly generated using Depth-First Search (DFS), ensuring unique environments for training.
- **Training Visualization**: Real-time plotting of the agent’s performance, including scores and mean scores over time.
- **Data Persistence**: Agent data and game state are saved to a SQL database for future use and analysis.
- **Game Simulation with Pygame**: Visualizes the maze and agent’s movements.

## Used Modules and Their Purpose:

- **Torch**: Provides the framework for implementing Deep Q-Learning. It is used for building and training neural networks, handling tensor operations, and performing backpropagation and optimization.
- **Pygame**: Facilitates the creation of the graphical user interface for visualizing the maze and the agent’s movements. It is used for rendering graphics and handling user inputs.
- **Matplotlib**: Used for plotting and visualizing the training process, including agent performance metrics such as scores and average scores over time.
- **Azure SQL**: Manages data persistence, allowing for the storage and retrieval of agent data and game states. It ensures that data is saved securely and can be accessed for future analysis.

## Examples 


https://github.com/user-attachments/assets/9b874013-224d-4c5b-ae8d-4948950452d2

## Database 

The database is created and managed in Azure SQL, providing a scalable and secure environment for storing and retrieving data.

- **Maze Data Storage**: The database saves the generated mazes, including their dimensions, layout, and start/end positions. This allows for the preservation of maze configurations for future reference or analysis.
- **Agent Data Storage**: It also stores data related to trained agents, including their performance metrics, model parameters, and training history. This ensures that information about each agent’s learning progress and results is retained for ongoing evaluation and improvement.
Azure SQL’s capabilities ensure that all data is efficiently managed, with automatic scaling and high availability features supporting the application’s needs.

## Example plot
![Figure_1](https://github.com/user-attachments/assets/bd5774e3-63bb-4a63-a4af-8f03e733f70f)

