# ===========> A module to visualise a DNN model <===========

import pygame
import pygame.gfxdraw
from numpy import array, random, exp

# LINEWIDTH = 1
BLACK = array([0, 0, 0])
WHITE = array([255, 255, 255])
# YELLOW = array([255, 255, 0])
# BLUE = array([0, 255, 255])
CURVANCE = 0.7
COLOUR_BIAS = -0.5
MAX_NEURONS = 50

# These distribution curves define the colour of a weight connection depending on the weight value
sigmoid = lambda x: 1 / (1 + exp(-x * CURVANCE))
gaussian = lambda x: exp(-(x ** 2 / 2 * CURVANCE ** 2))
spectrum = lambda x: array([sigmoid(x - COLOUR_BIAS), gaussian(x), sigmoid(-x - COLOUR_BIAS)])


def init_visualisation(self, nradius = 10, screen = None, screen_size = None, screen_coords = None):
    self.visualised = 'independently' if screen == None else 'dependently'
    self.nradius = nradius

    if self.visualised == 'independently':
        size = self.width, self.height = 1024, 768
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption('Neural Network')
    elif self.visualised == 'dependently':
        assert screen_size != None
        assert screen_coords != None
        self.width, self.height = screen_size
        self.screen_coords = screen_coords
        self.main_screen = screen
        self.screen = pygame.Surface(screen_size)

    self.amount_of_neurons = tuple([min(number_of_neurons, MAX_NEURONS) for number_of_neurons in self.shape])
    # self.bias = tuple([0 if number_of_neurons < MAX_NEURONS else number_of_neurons // 2 - MAX_NEURONS // 2 for number_of_neurons in self.shape])
    self.coordinates = []
    for i in range(self.layers):
        x = (i + 1) * self.width // self.layers - self.width // self.layers // 2
        self.coordinates.append(
            array([
                [x, y - self.height // self.amount_of_neurons[i] // 2] for y in range(1, self.height + 1) if y % (self.height // self.amount_of_neurons[i]) == 0
            ])
        )


def update_screen(self):
    if not self.visualised: return
    
    self.screen.fill(BLACK)
    for l in range(self.layers - 1):
        for i in range(self.amount_of_neurons[l]):
            for j in range(self.amount_of_neurons[l + 1]):
                # colour = (YELLOW  if self.weights[l][j, i] >= 0 else BLUE) * sigmoid(abs(self.weights[l][j, i]))
                colour = WHITE * spectrum(self.weights[l][j, i])
                pygame.draw.aaline(self.screen, colour, self.coordinates[l][i], self.coordinates[l + 1][j])
    for l in range(self.layers):
        for n in range(self.amount_of_neurons[l]):
            colour = WHITE * (self.activations[l][n] if 0 <= self.activations[l][n] <= 1 else sigmoid(self.activations[l][n]))
            pygame.gfxdraw.filled_circle(self.screen, *self.coordinates[l][n], self.nradius, colour)
            pygame.gfxdraw.aacircle(self.screen, *self.coordinates[l][n], self.nradius, WHITE)
    
    # pygame.draw.aaline(self.screen, WHITE, (random.randint(0, self.width), random.randint(0, self.height)), (random.randint(0, self.width), random.randint(0, self.height)))
    
    if self.visualised == 'dependently':
        self.main_screen.blit(self.screen, self.screen_coords)
    elif self.visualised == 'independently':        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.visualised = False
                
    pygame.display.update()
