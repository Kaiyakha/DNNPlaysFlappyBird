# ===========> A module to define entities of the Flappy Bird Game <===========

import os, random, itertools
import pygame

SCREENSIZE = SWIDTH, SHEIGHT = 1270, 768
IMAGE_PATH = os.path.dirname(__file__) + "/Images/"
BACKGROUND_IMAGE = pygame.image.load(IMAGE_PATH + "Background.png")
BACKGROUND_IMAGE = pygame.transform.scale(BACKGROUND_IMAGE, SCREENSIZE)
LOWER_PIPE_IMAGE = pygame.image.load(IMAGE_PATH + "Pipe.png")
LOWER_PIPE_IMAGE = pygame.transform.scale(LOWER_PIPE_IMAGE, (100, 450))
UPPER_PIPE_IMAGE = pygame.transform.flip(LOWER_PIPE_IMAGE, False, True)
BIRD_IMAGES = tuple([
    pygame.image.load(IMAGE_PATH + f"bird_{i}.png") for i in range(1, 4)
])
BIRD_IMAGES = itertools.cycle(itertools.chain(BIRD_IMAGES, BIRD_IMAGES[-2:0:-1]))

PIPE_BIAS_RANGE = 300
YELLOW = 255, 255, 0


class Bird:
    def __init__(self):
        self.body = next(BIRD_IMAGES).get_rect(center = (100, random.randint(SHEIGHT // 10, SHEIGHT - SHEIGHT // 10)))
        self.speed_y = 0
        self.g = 2
        self.score = 0

    def move(self, up = False):
        if up: self.speed_y = -20
        else: self.speed_y += self.g
        self.body.move_ip(0, self.speed_y)    
    
    def collided(self, pipe):
        if self.body.colliderect(pipe.lower_pipe) \
        or self.body.colliderect(pipe.upper_pipe) \
        or self.body.bottom > SHEIGHT - 100 \
        or self.body.top < 0:
            return True

    def draw(self, screen):
        image = pygame.transform.rotate(next(BIRD_IMAGES), -0.75 * self.speed_y)
        screen.blit(image, self.body)


class Pipe:
    register = []
    speed = -10

    def __init__(self):
        bias = random.randint(0, PIPE_BIAS_RANGE)
        self.lower_pipe = LOWER_PIPE_IMAGE.get_rect(bottomleft = (SWIDTH, SHEIGHT + bias))
        self.upper_pipe = UPPER_PIPE_IMAGE.get_rect(topleft = (SWIDTH, -PIPE_BIAS_RANGE + bias))
        Pipe.register.append(self)

    @classmethod
    def move(cls):
        for pipe in cls.register:
            pipe.lower_pipe.move_ip(cls.speed, 0)
            pipe.upper_pipe.move_ip(cls.speed, 0)

    @classmethod
    def draw(cls, screen):
        for pipe in cls.register:
            screen.blit(LOWER_PIPE_IMAGE, pipe.lower_pipe)
            screen.blit(UPPER_PIPE_IMAGE, pipe.upper_pipe)

    @classmethod
    def nearest(cls, pos):
        return next(pipe for pipe in cls.register if pos < pipe.lower_pipe.right)


class Text:
    def __init__(self, text, x, y, font):
        self.text_rendered = font.render(str(text), True, YELLOW)
        self.text_rendered_rect = self.text_rendered.get_rect(center = (x, y))
        self.text_rendered_rect.center = x, y

    def update(self, text, font):
        self.text_rendered = font.render(str(text), True, YELLOW)

    def draw(self, screen):
        screen.blit(self.text_rendered, self.text_rendered_rect)