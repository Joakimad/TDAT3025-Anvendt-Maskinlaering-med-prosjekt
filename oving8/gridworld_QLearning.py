import math
import random

import numpy as np

from oving8 import gridworld_environment

# VARIABLES
training_episodes = 500
run_episodes = 3

learning_rate = 0
exploration_rate = 0
min_learning_rate = 0.001
min_exploration_rate = 0.1

discount = 1
decay = 25


# Chooses if a known action or a random action will be taken.
def select_action(state_value, er):
    rand = np.random.random()
    if rand < er:
        return random.randint(0, env.action_space - 1)
    else:
        return np.argmax(q_table[state_value])


# Decays the learning rate so that the model learns more at the beginning.
def decay_rate(n, min_value):
    return max(min_value, min(1.0, 1.0 - math.log10((n + 1) / decay)))


# Get Game environment
env = gridworld_environment.Gridworld()

# Initial Q-table
q_table = np.zeros((10, 10) + (env.action_space,))

# Training
print("Training Started...")
for episode in range(training_episodes):

    print("Episode %s" % episode)
    current_state = env.reset()

    # Decay. Uses episode number as variable.
    learning_rate = decay_rate(episode, min_learning_rate)
    exploration_rate = decay_rate(episode, min_exploration_rate)
    # print("ER: %s" % exploration_rate)
    done = False
    while not done:
        # Decide known action or random action.
        action = select_action(current_state, exploration_rate)

        # Increment environment
        properties, reward, done, info = env.step(action)
        new_state = properties

        # Update Q-Table
        future_optimal_value = np.max(q_table[new_state])
        learned_value = reward + discount * future_optimal_value
        old_value = q_table[current_state][action]
        q_table[current_state][action] = (1 - learning_rate) * old_value + learning_rate * learned_value

        # Update state
        current_state = new_state
        # env.render(q_table)
    #print("Steps taken: %dt" % env.steps_taken)

print("Training Complete!")
input("Ready to start...\n")

# Run
for episode in range(run_episodes):
    print("Episode %s" % episode)
    current_state = env.reset()
    done = False
    while not done:
        # Decide known action or random action.
        action = select_action(current_state, exploration_rate)

        # Increment environment
        properties, reward, done, info = env.step(action)
        new_state = properties
        current_state = new_state

        # Render the cartpole environment
        env.render(q_table)
    print("Steps taken: %dt" % env.steps_taken)
