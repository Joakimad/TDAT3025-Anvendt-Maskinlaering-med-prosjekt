import pygame
import numpy as np
import random
import math as math

GRIDSIZE = 6
BOX_LENGTH = 500
GRID_WIDTH = BOX_LENGTH / GRIDSIZE


class Goal():
    def __init__(self, x=3, y=3):
        super().__init__()
        self.x = x
        self.y = y
        self.color = (0, 255, 0)


class Piece():
    def __init__(self, x=0, y=0):
        super().__init__()
        self.x = x
        self.y = y
        self.color = (0, 0, 255)

    def move(self, direction):
        if (direction == 0 and self.y > 0):
            self.y = self.y - 1
        elif (direction == 1 and self.y < GRIDSIZE - 1):
            self.y = self.y + 1
        elif (direction == 2 and self.x > 0):
            self.x = self.x - 1
        elif (direction == 3 and self.x < GRIDSIZE - 1):
            self.x = self.x + 1


class PlayerMoves:
    def __init__(self):
        self.moves = [0, 1, 2, 3]  # [up, down, left, right]

    def explore(self):
        return random.choice(self.moves)


class GridGame:
    def __init__(self):
        self.steps = 0
        self.piece = Piece()
        self.goal = Goal()

        self.playerMoves = PlayerMoves()

        self.display_screen = pygame.display.set_mode((BOX_LENGTH, BOX_LENGTH))
        pygame.display.set_caption("GridGame")

    def reset(self):
        self.piece = Piece()
        self.steps = 0
        return tuple([self.piece.x, self.piece.y])

    def step(self, action):
        self.steps = self.steps + 1
        self.piece.move(action)
        reward = 0
        if (self.piece.x == self.goal.x and self.piece.y == self.goal.y):
            reward = 1
        done = reward == 1
        return tuple([self.piece.x, self.piece.y]), reward, done

    def renderBox(self, x, y, color):
        surface = pygame.Surface((GRID_WIDTH, GRID_WIDTH))
        rectangle = surface.get_rect(
            center=(int(GRID_WIDTH / 2 + x * GRID_WIDTH), int(GRID_WIDTH / 2 + y * GRID_WIDTH)))
        pygame.draw.rect(self.display_screen, color, rectangle)

    def renderGoal(self):
        self.renderBox(self.goal.x, self.goal.y, self.goal.color)

    def renderPiece(self):
        self.renderBox(self.piece.x, self.piece.y, self.piece.color)

    def renderQ(self, Q_table):
        maxDigit = 0
        for i in Q_table:
            for j in i:
                if (np.argmax(j) > maxDigit): maxDigit = np.argmax(j)
        counti = 0
        for i in Q_table:
            countj = 0
            for j in i:
                colorValue = math.floor((np.argmax(j) * 255 / maxDigit))
                color = (colorValue, 0, 0)
                # print(color)
                self.renderBox(counti, countj, color)
                countj = countj + 1
            counti = counti + 1

    def render(self, Q_table):
        self.display_screen.fill((0, 0, 0))
        self.renderQ(Q_table)
        self.renderPiece()
        self.renderGoal()

        pygame.display.update()
        pygame.time.wait(40000)


if __name__ == "__main__":
    num_episodes = 1000
    max_steps_per_episode = 100
    learning_rate = 0.01
    discount_rate = 0.09

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.001
    exploration_rate_decay = 0.01

    env = GridGame()
    q_table = np.zeros((GRIDSIZE, GRIDSIZE) + (4,))
    print(q_table.shape)
    print(q_table)
    for episode in range(num_episodes):
        state = env.reset()

        done = False

        for step in range(max_steps_per_episode):
            exploration_rate_random = random.uniform(0, 1)
            if (exploration_rate_random > exploration_rate):
                action = np.argmax(q_table[state])
            else:
                action = env.playerMoves.explore()
            new_state, reward, done = env.step(action)
            # print(state, action)
            # print(q_table[state, :])
            q_table[state][action] = q_table[state][action] * (1 - learning_rate) + \
                                     learning_rate * (reward + discount_rate * np.max(q_table[new_state]))

            state = new_state
            if done:
                print("done")
                break
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * \
                           np.exp(-exploration_rate_decay * episode)
        print(episode, step, exploration_rate)
        # print(q_table)

    print(q_table)
    env.reset()
    input("...")
    env.render(q_table)
