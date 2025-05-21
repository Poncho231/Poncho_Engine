import cupy as np
import chess

convert = {
    "p": 1,    "b": 2,  "n": 3,  "r": 4,  "q": 5,  "k": 6,
    "P": 7,    "B": 8,  "N": 9,  "R": 10, "Q": 11, "K": 12,
    ".": 0
}
game = []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sig_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

sig_derivative_v = np.vectorize(sig_derivative)

def convert_board(board):
    symbols = str(board).split()
    return [convert[s] for s in symbols] # We're returning a CPU array to avoid too many H2D transfers


# tanh, ReLU, tanh
non_linears = [
    np.tanh,
    lambda x: np.maximum(0, x)
]
derivatives  = [
    lambda x: 1.0 - np.tanh(x)**2,
    lambda x: (x > 0).astype(float)
]
rs = np.random.default_rng()
weights = rs.uniform(-0.2,0.2,(6,64,64))
weights = np.asarray(weights)
biases = np.random.uniform(-0.2, 0.2, (6, 64))
result = 1


def eval(weights, biases, current_input_activation):
    """Return z_list, a_list, and final scalar output."""
    x = current_input_activation # x will be updated through layers
    z_list, a_list = [], [x]

    for i in range(6):
        z = x @ weights[i] + biases[i]
        z_list.append(z)
        x = non_linears[i % 2](z) # x is updated here
        a_list.append(x)

    # x now holds the activations of the last layer
    y_hat = sigmoid(np.sum(x))  # Use the final 'x'
    return z_list, a_list, y_hat


def gradient():
    global weights
    global biases
    loss_list = []  # we will populate this inside the loop below
    positions = []

    weight_grad_l = np.zeros_like(weights)
    bias_grad_l = np.zeros_like(biases)

    total_loss = 0.0
    for i in game:
        positions.append(convert_board(i))
    for position in np.array(positions):
        z_list, s_list, input_v = eval(weights, biases, position)

        delta_cur = input_v - result # Derivative of the loss function with respect to the given sum
        delta_cur *= sig_derivative(np.sum(s_list[-1]))  # chain rule through the sigmoid(sum)

        delta_cur = np.ones(64) * delta_cur # THat's the derivative of the sum function

        for layer in reversed(range(6)):
            dz = delta_cur * derivatives[layer % 2](z_list[layer])  # scalar multiplication
            weight_grad_l[layer] += np.outer(s_list[layer], dz)
            bias_grad_l[layer] += dz
            delta_cur = weights[layer].T @ dz

    loss_list.append(total_loss)
    return weight_grad_l, bias_grad_l  # same shapes as weights / biases




board = chess.Board()
test_board = board.copy()
turn = 1
for j in range(100):
    turn = 1
    while not board.is_game_over():
        best_move = None

        if turn == 1:  # White's turn
            # White wants to find a move that MINIMIZES Black's score
            current_best_score_for_black = float('inf')
        else:  # Black's turn (turn == -1)
            # Black wants to find a move that MAXIMIZES Black's score
            current_best_score_for_black = -float('inf')

        possible_moves = list(board.legal_moves)
        test_board = board.copy()
        possible_moves_full_cpu = []
        for i in possible_moves:
            test_board.push(i)
            possible_moves_full_cpu.append(convert_board(test_board))
            test_board = board.copy()
        possible_moves_full_gpu = np.array(possible_moves_full_cpu)

        for index, vector in enumerate(possible_moves_full_gpu):
            value = list(eval(weights, biases, vector))[-1]
            if turn == 1:  # White's turn
                if value < current_best_score_for_black:
                    current_best_score_for_black = value
                    best_move = possible_moves[index]
            else:  # Black's turn (turn == -1)
                if value > current_best_score_for_black:
                    current_best_score_for_black = value
                    best_move = possible_moves[index]
        game.append(board.copy())
        board.push(best_move)
        turn = -turn  # Flip turn for the next player
    game.append(board.copy())
    result = board.result()
    if result[2] == "1":
        result = 0
    elif result[2] == "0":
        result = 1
    else:
        result = 0.5

    weight_grad, bias_grad = gradient()
    weights -= weight_grad * 0.01 / len(game)
    biases -= bias_grad * 0.01 / len(game)
    board = chess.Board()
    game.clear()

while not board.is_game_over():
    best_move = None

    if turn == 1:  # White's turn
        # White wants to find a move that MINIMIZES Black's score
        current_best_score_for_black = float('inf')
    else:  # Black's turn (turn == -1)
        # Black wants to find a move that MAXIMIZES Black's score
        current_best_score_for_black = -float('inf')

    possible_moves = list(board.legal_moves)
    possible_moves_full_cpu = []
    for i in possible_moves:
        test_board.push(i)
        possible_moves_full_cpu.append(convert_board(test_board))
        test_board = board.copy()
    possible_moves_full_gpu = np.array(possible_moves_full_cpu)

    for index, vector in enumerate(possible_moves_full_gpu):
        value = list(eval(weights, biases, vector))[-1]
        if turn == 1:  # White's turn
            if value < current_best_score_for_black:
                current_best_score_for_black = value
                best_move = possible_moves[index]
        else:  # Black's turn (turn == -1)
            if value > current_best_score_for_black:
                current_best_score_for_black = value
                best_move = possible_moves[index]
    game.append(board.copy())
    board.push(best_move)
    turn = -turn  # Flip turn for the next player
game.append(board.copy())