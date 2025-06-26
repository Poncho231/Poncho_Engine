import numpy as np
import chess
import os
import numpy as np_cpu
import cProfile
import pstats

import random
from multiprocessing import Pool, cpu_count
from main import make_game
from main import save_games
from main import gradient
from main import gradient_worker

cores = cpu_count()
with Pool(cores) as pool:
    gameList = pool.map(make_game, range(10000))
save_games([game[0] for game in gameList], [game[1] for game in gameList])