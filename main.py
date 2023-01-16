import pygame
import numpy as np
import math
import random

from utils import *

# colors
BLACK = np.array((0, 0, 0))
WHITE = np.array((255, 255, 255))
RED = np.array((255, 0, 0))
GREEN = np.array((0, 255, 0))
BLUE = np.array((0, 0, 255))

pygame.init()

FPS = float("inf")
SIZE = (800, 800)
WIN = pygame.display.set_mode(SIZE)

pygame.display.set_caption("Ball Elastic Collision")


def distance(p1, p2):
    dp = p2 - p1
    return math.sqrt(np.sum(dp**2))


def collide(b1, b2):
    return distance(b1.pos, b2.pos) <= b1.radius + b2.radius


def elatic_collision(v1i, m1, v2i, m2):
    """
    Returns final velocity for an elastic collision.

    Parameters:
        v1i (np.ndarray): inital velocity of object 1
        m1 (int): mass of object 1
        v2i (np.ndarray): inital velocity of object 2
        m2 (int): mass of object 2

    Returns:
        vf (tuple): final velocity of object 1 and 2
    """

    v2f = (2 * m1 * v1i + m2 * v2i - m1 * v2i) / (m1 + m2)
    v1f = v2i + v2f - v1i

    return v1f, v2f


class Ball:
    deltatime = FPS

    def __init__(
        self,
        pos: np.ndarray,
        radius: float,
        color: np.ndarray,
        inital_veloctiy: np.ndarray,
        mass: int,
    ):
        self.pos = pos
        self.radius = radius
        self.color = color
        self.velocity = inital_veloctiy
        self.mass = mass

        self.surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA, 32)
        pygame.draw.circle(self.surface, color, (radius, radius), radius)
        self.text = Text(
            (self.radius, self.radius),
            str(mass),
            int(self.radius * 0.9),
            color=(255 - color),
        )
        self.text.draw(self.surface)

    def update(self, obstacles):
        self.pos = self.pos + self.velocity / self.deltatime

        self.boundary_collision()

    def boundary_collision(self):
        if self.pos[0] - self.radius <= 0 or self.pos[0] + self.radius >= SIZE[0]:
            self.velocity[0] = -self.velocity[0]
        if self.pos[1] - self.radius <= 0 or self.pos[1] + self.radius >= SIZE[1]:
            self.velocity[1] = -self.velocity[1]

    def draw(self, win: pygame.Surface):
        win.blit(self.surface, self.pos - self.radius)

    @classmethod
    def update_deltatime(cls, deltatime):
        cls.deltatime = deltatime


def generate_balls(amt):
    balls = []
    for _ in range(amt):
        collided = True
        while collided:
            mass = random.randint(0, 1000)

            radius = random.randint(20, 40)
            pos = np.array(
                [
                    random.randint(math.ceil(0 + radius), math.floor(SIZE[0] - radius)),
                    random.randint(math.ceil(0 + radius), math.floor(SIZE[1] - radius)),
                ]
            )
            color = np.array(
                [
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ]
            )
            velocity = np.array([random.randint(0, 400), random.randint(0, 400)])

            ball = Ball(pos, radius, color, velocity, mass)

            for b in balls:
                if collide(b, ball):
                    break
            else:
                collided = False

        balls.append(ball)

    # balls.append(Ball(np.array([100, 100]), 40, RED, np.array((10, 10)), 1))  # TEST

    return balls


def generate_unique_balls(balls):
    unique_set = set()
    for b1 in range(len(balls)):
        for b2 in range(len(balls)):
            if b1 == b2:
                continue
            sorted_tuple = tuple(sorted((b1, b2)))
            unique_set.add((balls[sorted_tuple[0]], balls[sorted_tuple[1]]))
    unique_balls = list(unique_set)
    return unique_balls


def draw(win, balls, fps_text):
    win.fill(WHITE)

    for ball in balls:
        ball.draw(win)

    fps_text.draw(WIN)

    pygame.display.update()


def main():
    BALL_AMOUNT = 10
    balls = generate_balls(BALL_AMOUNT)

    unique_balls = generate_unique_balls(balls)

    fps_text = Text((SIZE[0] * 0.9, 20), "FPS: " + str(FPS), 30)

    run = True
    clock = pygame.time.Clock()
    while run:
        time_passed = clock.tick(FPS)
        fps = clock.get_fps()
        fps_text.set_text("FPS: " + str(int(fps)))
        Ball.update_deltatime(fps if fps > 0 else FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    run = False
                    break
                if event.key == pygame.K_r:
                    balls = generate_balls(BALL_AMOUNT)
                    unique_balls = generate_unique_balls(balls)

        for ball in balls:
            ball.update([b for b in balls if b != ball])

        for b1, b2 in unique_balls:
            if collide(b1, b2):
                b1.velocity, b2.velocity = elatic_collision(
                    b1.velocity, b1.mass, b2.velocity, b2.mass
                )

        draw(WIN, balls, fps_text)

    pygame.quit()


if __name__ == "__main__":
    main()
