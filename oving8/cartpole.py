from typing import Tuple

import gym
import math
import numpy as np
import time
from sklearn.preprocessing import KBinsDiscretizer

# VARIABLES
training_episodes = 1000
run_episodes = 3

learning_rate = 0
exploration_rate = 0
min_learning_rate = 0.1
min_exploration_rate = 0.1

discount = 1
decay = 20


# FUNCTIONS


# Continuous to Discrete. Q learning demands discrete states, but this problem has continuous states.
def discretizer(_, __, angle, pole_velocity) -> Tuple[int, ...]:
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))


# Chooses if a known action or a random action will be taken.
def select_action(state_value, er):
    if np.random.random() < er:
        print("Random Action")
        return env.action_space.sample()
    else:
        print("Educated Action")
        return np.argmax(Q_table[state_value])


# Decays the learning rate so that the model learns more at the beginning.
def decay_rate(n, min_value):
    return max(min_value, min(1.0, 1.0 - math.log10((n + 1) / decay)))





# EXECUTABLE


# Get cartpole environment
env = gym.make("CartPole-v0")
env.reset()

# Define bins
bins = (6, 12)
lower_bounds = [env.observation_space.low[2], -math.radians(50)]
upper_bounds = [env.observation_space.high[2], math.radians(50)]

# Initial Q-table
Q_table = np.zeros(bins + (env.action_space.n,))
print(Q_table.shape)


# Training
print("Training Started...")
for episode in range(training_episodes):
    current_state = discretizer(*env.reset())

    # Decay. Uses episode number as variable.
    learning_rate = decay_rate(episode, min_learning_rate)
    exploration_rate = decay_rate(episode, min_exploration_rate)

    done = False
    while not done:
        # Decide known action or random action.
        action = select_action(current_state, exploration_rate)

        # Increment environment
        properties, reward, done, _ = env.step(action)
        print(properties, reward, done)
        new_state = discretizer(*properties)
        print(new_state)

        # Update Q-Table
        future_optimal_value = np.max(Q_table[new_state])
        learned_value = reward + discount * future_optimal_value
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1 - learning_rate) * old_value + learning_rate * learned_value

        current_state = new_state

        # env.render()

print("Training Complete!")
input("Ready to start...\n")

# Run
for episode in range(run_episodes):
    current_state = discretizer(*env.reset())
    done = False
    start = time.time()
    while not done:
        # Decide known action or random action.
        action = select_action(current_state, exploration_rate)

        # Increment environment
        properties, reward, done, _ = env.step(action)
        new_state = discretizer(*properties)
        current_state = new_state

        # Render the cartpole environment
        env.render()

    # Print time
    end = time.time()
    print("Time %s" % (end - start))
