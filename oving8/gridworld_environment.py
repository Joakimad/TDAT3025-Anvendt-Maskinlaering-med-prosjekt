import math
import sys

import numpy as np
import pygame

# VARIABLES
BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
BLUE = (30, 144, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

WINDOW_SIZE = 400
BLOCS_PER_ROW = 10
BLOCK_SIZE = WINDOW_SIZE / BLOCS_PER_ROW
PLAYER_SIZE = BLOCK_SIZE
GOAL_X = 2
GOAL_Y = 2
TURN_TIME = 200


# CLASSES
class Player:
    def __init__(self):
        self.x = 4
        self.y = 6
        self.color = BLUE

    def move(self, direction):
        # UP
        if direction == 0 and self.y > 0:
            self.y -= 1
        # DOWN
        elif direction == 1 and self.y < BLOCS_PER_ROW - 1:
            self.y += 1
        # LEFT
        elif direction == 2 and self.x > 0:
            self.x -= 1
        # RIGHT
        elif direction == 3 and self.x < BLOCS_PER_ROW - 1:
            self.x += 1


class Goal:
    def __init__(self):
        self.x = GOAL_X
        self.y = GOAL_Y
        self.color = RED


class Gridworld:
    def __init__(self):
        self.player = Player()
        self.goal = Goal()
        self.action_space = 4
        self.state = None
        self.steps_taken = 0
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))

    def step(self, action):
        self.steps_taken += 1
        # print("Step: %s - Action %s" % (self.steps_taken, action))
        self.player.move(action)
        self.state = (self.player.x, self.player.y, self.goal.x, self.goal.y)

        reward = 0
        done = False
        info = {"direction": action}
        if self.player.x == self.goal.x and self.player.y == self.goal.y:
            reward = 1
            done = True

        return np.array(self.state), reward, done, info

    def reset(self):
        self.player = Player()
        self.goal = Goal()
        self.state = (self.player.x, self.player.y, self.goal.x, self.goal.y)
        self.steps_taken = 0

        return np.array(self.state)

    def draw_grid(self):
        for row in range(BLOCS_PER_ROW):
            for col in range(BLOCS_PER_ROW):
                rect = pygame.Rect(BLOCK_SIZE * col,
                                   BLOCK_SIZE * row,
                                   BLOCK_SIZE,
                                   BLOCK_SIZE)
                pygame.draw.rect(self.screen, WHITE, rect, 1)

    def draw_rect(self, x, y, color):
        rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, PLAYER_SIZE, PLAYER_SIZE)
        pygame.draw.rect(self.screen, color, rect)

    def render_table(self, q_table):
        for y in range(len(q_table)):
            for x in range(len(q_table[y])):
                if np.argmax(q_table[x][y][self.goal.x][self.goal.y]) > 0:
                    self.draw_rect(x, y, YELLOW)

    def draw_highest_Q_value(self, q_table):
        max_digit = 2
        for i in q_table:
            for j in i:
                if np.argmax(j) > max_digit:
                    max_digit = np.argmax(j)

        neighbour_x, neighbour_y = self.player.x, self.player.y
        # UP
        if max_digit == 0 and self.player.y > 0:
            neighbour_y = self.player.y - 1
        # DOWN
        elif max_digit == 1 and self.player.y < BLOCS_PER_ROW - 1:
            neighbour_y = self.player.y + 1
        # LEFT
        elif max_digit == 2 and self.player.x > 0:
            neighbour_x = self.player.x - 1
        # RIGHT
        elif max_digit == 3 and self.player.x < BLOCS_PER_ROW - 1:
            neighbour_x = self.player.x + 1
        self.draw_rect(neighbour_x, neighbour_y, YELLOW)

    def render(self, q_table):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Reset backdrop
        self.screen.fill(BLACK)
        self.draw_grid()

        # Q_table values
        # self.draw_highest_Q_value(Q_table)
        # self.render_table(q_table)

        # Draw Player & Target
        self.draw_rect(self.goal.x, self.goal.y, self.goal.color)
        self.draw_rect(self.player.x, self.player.y, self.player.color)
        pygame.display.update()
        pygame.time.wait(TURN_TIME)
