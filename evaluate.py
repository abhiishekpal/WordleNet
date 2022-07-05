
import torch
from wordle_simulator import Wordle
from model import DQNet
from play import Agent
import numpy as np

import torch
import tqdm
 

env = Wordle()
input_dimension, output_dimension = env.size, env.ACTION_SPACE_SIZE

agent = Agent(env.size, env.ACTION_SPACE_SIZE)
model = DQNet(input_dimension, output_dimension)
model.load_state_dict(torch.load('./model2.pth'))
model.eval()

done = False

step = 1
with torch.no_grad():
    
    current_state = env.reset()
    while not done:

        sort_action = np.argsort(agent.get_qs(current_state).cpu().numpy())[0]
        for i in range(len(sort_action)):
            action = sort_action[i]
            new_state, reward, done = env.step(action)
            if done == None:
                done = False
                continue
            else:
                break
        
        current_state = new_state
        step += 1
        if step==7:
            print(reward)
            break

    