# ===========> The main module to run the game <===========

# Here a deep neural network model learns to play the Flappy Bird game
# The game is run over and over again until the model can handle the bird correctly

# It is much easier to train a DNN model to play this game than one may think
# The only input a model needs is the vertical distance to the gap between the nearest pipes
# It does not matter how far from this gap the bird is along the horizontal axis
# The model simply learns to align the bird with the gap between the nearest pipes


from Entities import *
from NN import *

DELAY = 5
PIPE_FREQUENCY = 40
SCORES_POS = 20, 20
FLOOR_HEIGHT = 100

screen = pygame.display.set_mode(SCREENSIZE)
pygame.display.set_caption("Flappy Bird")
pygame.font.init()
font = pygame.font.Font(None, 48)


def play(nn):
    Pipe()
    bird = Bird()
    scores = Text(bird.score, *SCORES_POS, font)

    run = True; iterations = 0
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit()

        pygame.time.delay(DELAY)
        up = False

        # Find the nearest pipe and ask the model whether to move upwards or not
        nearest_pipe = Pipe.nearest(bird.body.left)
        X = [(nearest_pipe.upper_pipe.bottom + nearest_pipe.lower_pipe.top) // 2
           + PIPE_BIAS_RANGE// 7 - bird.body.center[1]]
        nn.forward_prop(X)
        up = round(nn.activations[-1][0])

        # Add a new pipe to the register if the last pipe has moved far enough
        iterations += 1
        if not iterations % PIPE_FREQUENCY:
            Pipe()
            iterations = 0

        screen.blit(BACKGROUND_IMAGE, (0, 0))
        bird.move(up)
        bird.draw(screen)
        Pipe.move()
        Pipe.draw(screen)

        # If the bird has collided, the game is over and data for training is collected
        if bird.collided(nearest_pipe):
            run = False
            train_x = X
            if nearest_pipe.lower_pipe.top - PIPE_BIAS_RANGE // 7 > bird.body.center[1]: train_y = 0
            else: train_y = 1
        elif nearest_pipe.lower_pipe.right < bird.body.left:
            bird.score += 1
            scores.update(bird.score, font)

        if Pipe.register[0].lower_pipe.right < 0: del Pipe.register[0]
        scores.draw(screen)

        if nn.visualised: nn.update_screen()
        else: pygame.display.update()

    Pipe.register.clear()

    return train_x, train_y


nn = NeuralNetwork((1, 2, 3, 1))
nn.init_visualisation(7, screen, (200, SHEIGHT), (SWIDTH - 200, 0))
for i in range(10000):
    X, Y = play(nn)
    nn.train([X], [Y], 0.8, 1)