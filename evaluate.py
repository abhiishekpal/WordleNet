
import torch
from wordle_simulator import Wordle
from model import DQNet
from play import Agent
import numpy as np

import torch
import tqdm
import random
 

env = Wordle()
input_dimension, output_dimension = env.size, env.ACTION_SPACE_SIZE

agent = Agent(env.size, env.ACTION_SPACE_SIZE)
model = DQNet(input_dimension)
model.load_state_dict(torch.load('./wordle-densev4.0.pt'))

model.eval()

done = False

step = 1

def reshape_array(arr):

    return arr
    temp = arr.reshape(*arr.shape[:-2], -1)
    temp = np.expand_dims(temp, axis=0)
    return temp


with torch.no_grad():
    
    current_state = env.reset()
    visited_word = set()
    print(env.goal_word)
    print('----')
    subspace = list(set(env.CANDIDATE_SPACE))[:20]
    while not done:

        scores = []
        subspace = list(set(subspace).difference(visited_word))
        random.shuffle(subspace)
        subspace2 = list(set(subspace + [env.goal_word]))
        random.shuffle(subspace2)

        for cand in (subspace2):
            temp_state = env.create_state(cand)
            print(temp_state)
            cs = reshape_array(temp_state)
            scores.append(agent.get_qs(cs).cpu().numpy())
            print(cand, '----> ', scores[-1])
        action = np.argmax(scores)
        
        sel_word = list(subspace2)[action]
        visited_word.add(sel_word)

        print(sel_word, step)
        new_state, _, _ = env.step(sel_word)
        current_state = new_state.copy()
        if sel_word==env.goal_word:
            break
        step += 1
        if step==7:
            break

    