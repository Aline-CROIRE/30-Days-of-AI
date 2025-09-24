# --- 1. Import Libraries ---
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from collections import deque
import numpy as np
import random
import matplotlib.pyplot as plt
import os

print("Libraries imported successfully!")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Gymnasium Version: {gym.__version__}")

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --- 2. Define the DQN Agent Class ---
class DQNAgent:
    """A Deep Q-Network Agent for Reinforcement Learning."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Experience Replay: store past experiences in a deque
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters for the learning algorithm
        self.gamma = 0.95    # Discount rate for future rewards
        self.epsilon = 1.0   # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Build two neural networks: one for learning, one for predicting target values
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Builds the Neural Network for approximating Q-values."""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copies the weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Chooses an action based on the current state using an epsilon-greedy policy.
        With probability epsilon, it takes a random action (exploration).
        Otherwise, it takes the best action predicted by the Q-network (exploitation).
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Trains the main Q-network using a random minibatch of experiences from memory."""
        if len(self.memory) < batch_size:
            return # Don't train if memory is smaller than batch size
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        # Predict Q-values for the next states using the stable target model
        target_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Calculate the target Q-value for each experience in the batch
        targets = rewards + self.gamma * (np.amax(target_q_values, axis=1)) * (1 - dones)

        # Get the current Q-value predictions from the main model
        current_q_values = self.model.predict(states, verbose=0)
        
        # Update the Q-value for the action that was actually taken
        for i, action in enumerate(actions):
            current_q_values[i][action] = targets[i]
        
        # Train the main model on the updated Q-values
        self.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon to shift from exploration to exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# --- 3. Setup and Run the Training Loop ---
print("\n--- Setting up the CartPole Environment ---")
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Training parameters
EPISODES = 500 # Set a lower number for quicker testing (e.g., 500)
BATCH_SIZE = 32

# Create our agent
agent = DQNAgent(state_size, action_size)
scores = []

print("\n--- Starting Training ---")
for e in range(EPISODES):
    state, info = env.reset()
    state = np.reshape(state, [1, state_size])
    score = 0
    
    # Each episode runs for a maximum of 500 time steps
    for time in range(500):
        # Uncomment the next line to watch the agent play in real-time (can be slow)
        # env.render()
        
        # Agent chooses an action
        action = agent.act(state)
        
        # Agent takes the action, environment returns the result
        next_state, reward, done, truncated, info = env.step(action)
        score += reward
        next_state = np.reshape(next_state, [1, state_size])
        
        # Agent stores this experience in its memory
        agent.remember(state, action, reward, next_state, done)
        
        # Move to the next state
        state = next_state
        
        if done or truncated:
            # Update the target network at the end of each episode
            agent.update_target_model()
            break
            
    scores.append(score)
    print(f"Episode: {e+1}/{EPISODES}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
    
    # Agent learns from its memories
    agent.replay(BATCH_SIZE)

print("\n--- Training Finished ---")

# --- 4. Visualize the Learning Process ---
plt.figure(figsize=(12, 6))
plt.plot(scores)
# Add a moving average to see the trend more clearly
moving_avg = np.convolve(scores, np.ones(50)/50, mode='valid')
plt.plot(moving_avg, label='50-Episode Moving Average', color='red')
plt.title("DQN Agent Performance on CartPole-v1")
plt.xlabel("Episode")
plt.ylabel("Score (Time Steps Pole Remained Up)")
plt.legend()
plt.grid(True)
plt.show()