from wordle_simulator import Wordle
from model_reinforce import Reinforce
import copy
from collections import deque
import random
import numpy as np
import tqdm
import torch.nn as nn
import torch
from torch.distributions import Categorical

import random
# random.seed(0)
from torch.utils.tensorboard import SummaryWriter




REPLAY_MEMORY_SIZE = 500
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 500
EPISODES = 100000

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__=='__main__':

    writer = SummaryWriter()
    env = Wordle()
    model = Reinforce(input_dim=130, output_dim=len(env.CANDIDATE_SPACE), hidden_dim=1000)
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    alpha = 1

    for episodes in tqdm.tqdm(range(EPISODES)):

        # tracking rewards for each game 
        episode_reward = 0

        # Reset the board state and empty the visited word list
        current_state = env.reset()
        

        # select a random candidate space from the large space of words for each game
        cand_space = list(env.CANDIDATE_SPACE)


        # starting stepping in the game
        step = 1
        
        episode_list = []
        while(1):
            cs = torch.tensor(current_state).to(device).type(torch.float32)
            cs = cs.flatten(start_dim=1)
            scores = model(cs)
            # action = torch.multinomial(scores, num_samples = 1)
            dist = Categorical(scores)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # create the vector for the chosen word
            current_selected_word = list(cand_space)[action]

            # if the chosen word is infact the goal word we update the memory with the states and reward, train a step and terminate
            if current_selected_word == env.goal_word:
                new_state, _ = env.step(current_selected_word, current_state)
                reward = 1
                episode_reward = reward
                episode_list.append((current_state, log_prob, reward))
                break

            new_state, reward = env.step(current_selected_word, current_state)
            episode_reward = np.mean(reward)
            done = 1 if step==6 else 0
            episode_list.append((current_state, log_prob, np.mean(reward)))
            current_state = new_state.copy()
            
            # break if we are exceeding 6 steps (which is also wordle rule)
            if step==6:
                break
            step += 1

        accumulated_reward = [0]*len(episode_list)
        accumulated_reward[-1] = episode_list[-1][2]*(DISCOUNT**(len(episode_list)-1))
        for i in range(len(episode_list)-2, -1, -1):
            accumulated_reward[i] = accumulated_reward[i+1] + (episode_list[i][2]*(DISCOUNT**i))
        optimizer.zero_grad()
        for i in range(len(episode_list)):
            loss = torch.mul(torch.log(episode_list[i][1]), accumulated_reward[i]*alpha)
            loss.backward()
        optimizer.step()
        
        if episodes%50==1:
            torch.save(model.state_dict(), "wordle-reinforce.pt")

        writer.add_scalar("Reward", episode_reward, episodes)
        writer.add_scalar("Steps Taken", step, episodes)