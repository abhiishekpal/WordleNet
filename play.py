
from wordle_simulator import Wordle
from model import DQNet
import copy
from collections import deque
import random
import numpy as np
import tqdm
import torch.nn as nn
import torch

import random
random.seed(0)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

REPLAY_MEMORY_SIZE = 500
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 64
DISCOUNT = 0.5
UPDATE_TARGET_EVERY = 100
EPISODES = 20000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self, input_dimension, output_dimension) -> None:
         
        self.model = DQNet(input_dimension)
        self.target_model = copy.deepcopy(self.model)
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-3)
        if torch.cuda.is_available():
            self.model.cuda()
            self.target_model.cuda()
        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        
        self.replay_memory.append(transition)


    def train(self, terminal_state):
        
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        # print(len(self.replay_memory) , MIN_REPLAY_MEMORY_SIZE, len(minibatch))
        # (current_state, reward, new_state, done, subspace, visited_word.copy(), step)

        self.model.eval()
        with torch.no_grad():    
            current_states = np.array([reshape_array(transition[0]) for transition in minibatch] )
            current_states = torch.tensor([item for item in current_states], dtype=torch.float)
            if torch.cuda.is_available():
                current_states = current_states.cuda()
            current_qs_list = self.model(current_states)
            
            future_qs_list = []
            for transition in minibatch:
                step = transition[6]
                if step==6:
                    step -= 1
                temp = []
                for all_words in transition[4]:
                    arr = transition[2].copy()
                    for i in range(len(all_words)):
                        if all_words in transition[5]:
                            continue
                        arr[0][step][i] = (ord(all_words[i])-97) / 26 
                        arr[1][step][i] = 0
                        arr2 = np.array(arr)
                        temp.append(arr2)

                temp = np.array(temp)
                new_current_states = torch.tensor([reshape_array(item) for item in temp], dtype=torch.float)
                if torch.cuda.is_available():
                    new_current_states = new_current_states.cuda()

                future_qs_list.append(self.target_model(new_current_states))
            
        future_qs_list = [item.cpu().numpy() for item in future_qs_list]
        current_qs_list = current_qs_list.cpu().numpy()

        X = []
        y = []
        for index, (current_state, reward, new_current_state, done, cs, vw, st) in enumerate(minibatch):
            if not done and st<6:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = np.array(reward, dtype=np.float32)

            current_qs = new_q.reshape((1))

            X.append(reshape_array(current_state))
            y.append(current_qs)

        X = torch.tensor([itm for itm in X], dtype=torch.float)
        y = torch.tensor([itm for itm in y], dtype=torch.float)
        
        self.model.train()
        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        output = self.model(X)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            torch.save(self.model.state_dict(), 'model2.pth')
            self.target_model = copy.deepcopy(self.model)
            self.target_update_counter = 0

    def get_qs(self, state):
        self.model.eval()
        with torch.no_grad():
            state = torch.tensor([itm for itm in state], dtype=torch.float)
            state = state.unsqueeze(0)
            
            if torch.cuda.is_available():
                state = state.cuda()
            return self.model(state)


def reshape_array(arr):
    temp = arr.reshape(*arr.shape[:-2], -1)
    temp = np.expand_dims(temp, axis=0)
    return temp

def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):

    """Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
    Begin at 1. until warmup_steps steps have been taken; then
    Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
    Use epsilon from there on.
    Args:
    decay_period: float, the period over which epsilon is decayed.
    step: int, the number of training steps completed so far.
    warmup_steps: int, the number of steps taken before epsilon is decayed.
    epsilon: float, the final value to which to decay the epsilon parameter.
    Returns:
    A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus


if __name__=='__main__':

    env = Wordle()
    agent = Agent(env.size, env.ACTION_SPACE_SIZE)
    ep_rewards = [-200]

    total_step = 0

    for episodes in tqdm.tqdm(range(EPISODES)):
        # print(episodes)
        episode_reward = 0
        step = 1
        env = Wordle()
        current_state = env.reset()
        done = False
        iter = 0
        prev_reward = 0
        visited_word = set()
        epsilon, decay_period, warmup_steps = 0.3, 25000, 500
        print('--'*4)
        print(env.goal_word)
        print('--'*4)
        while not done:
            total_step += 1
            decaying_epsilon = linearly_decaying_epsilon(decay_period, total_step, warmup_steps, epsilon)
            subspace = list(set(env.CANDIDATE_SPACE).difference(visited_word))
            random.shuffle(subspace)
            subspace = list(set(subspace[:15] + [env.goal_word]))

            if np.random.random() > 0.1:#decaying_epsilon:
                scores = []
                visited_word2 = set()                
                for cand in (subspace):
                    visited_word2.add(cand)
                    for i in range(len(cand)):
                        current_state[0][step-1][i] = (ord(cand[i])-97) / 26 
                        current_state[1][step-1][i] = 0
                    cs = reshape_array(current_state)
                    scores.append(agent.get_qs(cs).cpu().numpy())
                action = np.argmax(scores)
            else:
                action = np.random.randint(0, len(subspace))
            

            sel_word = list(subspace)[action]
            visited_word.add(sel_word)

            for i in range(len(sel_word)):
                current_state[0][step-1][i] = (ord(sel_word[i])-97) / 26 
                current_state[1][step-1][i] = 0

            if sel_word == env.goal_word:
                reward, done = 1, 1
                new_state, _ , _ = env.step(sel_word)
                agent.update_replay_memory((current_state, reward, new_state, done, subspace, visited_word.copy(), step))
                agent.train(done)
                episode_reward += reward
                break

            new_state, reward, done = env.step(sel_word)

            if reward is None:
                continue
            
           

            rew = 0
            for itm in reward:
                if itm  == -1:
                    rew -= 1
                elif itm == 0:
                    rew -=0.1
                else:
                    rew += 1
            
          
            reward = rew/len(reward)
            
            print(sel_word, rew, reward)
            # if reward<0:
            #     reward = 0
            # else:
            #     reward = 1

            if episodes%250==1:
                torch.save(agent.model.state_dict(), "wordle2.pt")
            
            episode_reward += reward
              
            agent.update_replay_memory((current_state, reward, new_state, done, subspace, visited_word.copy(), step))
            agent.train(done)

            current_state = new_state
            
            step += 1
            if step==7:
                break

        print('--'*4)
        writer.add_scalar("Reward", episode_reward/step, episodes)
        writer.add_scalar("Steps Taken", step, episodes)
        ep_rewards.append(episode_reward)







    