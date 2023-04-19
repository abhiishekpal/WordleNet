
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
# random.seed(0)
from torch.utils.tensorboard import SummaryWriter


REPLAY_MEMORY_SIZE = 500
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 500
EPISODES = 30000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self, input_dimension, output_dimension) -> None:
        
        self.model = DQNet(input_dimension)
        self.target_model = copy.deepcopy(self.model)
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-4)
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
            current_states = np.array([fuse_array(transition[0], transition[1]) for transition in minibatch] )
            current_states = torch.tensor([current_states], dtype=torch.float)
            if torch.cuda.is_available():
                current_states = current_states.cuda()
            current_qs_list = self.model(current_states)
            
            future_qs_list = []
            for transition in minibatch:
                temp = []
                for all_words in transition[4]:
                    if all_words in transition[5]:
                            continue
                    new_state  = np.zeros(transition[0].shape)
                    for j, w in enumerate(all_words):
                        new_state[0][(ord(w)-97)][j] = 1
                    temp.append(new_state)

                temp = np.array(temp)
                new_current_states = torch.tensor([fuse_array(transition[0], item) for item in temp], dtype=torch.float)
                if torch.cuda.is_available():
                    new_current_states = new_current_states.cuda()

                future_qs_list.append(self.target_model(new_current_states))
            
        future_qs_list = [item.cpu().numpy() for item in future_qs_list]
        current_qs_list = current_qs_list.cpu().numpy()

        X = []
        y = []
        for index, (current_state, candidate_vec, reward, done, cs, vw, st) in enumerate(minibatch):
            if not done and st<6:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = np.array(reward, dtype=np.float32)

            current_qs = new_q.reshape((1))

            X.append(fuse_array(current_state, candidate_vec))
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


def fuse_array(arr1, arr2):

    arr1 = arr1.flatten()
    arr2 = arr2.flatten()

    arr = np.concatenate((arr1, arr2), axis=0)

    return arr


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

    writer = SummaryWriter()
    env = Wordle()
    agent = Agent((36, 28), env.ACTION_SPACE_SIZE)
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
        epsilon, decay_period, warmup_steps = 0.01, 5, 3
        # print('--'*4)
        # print(env.goal_word)
        # print('--'*4)
        cand_space = list(env.CANDIDATE_SPACE)
        decaying_epsilon = linearly_decaying_epsilon(decay_period, episodes, warmup_steps, epsilon)
        # print('decaying epsilon: ', decaying_epsilon)
        while not done:
            total_step += 1
            random.shuffle(cand_space)
            subspace = list(set(cand_space[:20]+[env.goal_word]).difference(visited_word))
            random.shuffle(subspace)
    
            if np.random.random() > decaying_epsilon:
                scores = []                
                for cand in (subspace):
                    candidate_vec = env.get_candidate_vec(cand)
                    cs = fuse_array(current_state, candidate_vec)
                    scores.append(agent.get_qs((cs)).cpu().numpy())
                action = np.argmax(scores)
            else:
                action = np.random.randint(0, len(subspace))
            

            sel_word = list(subspace)[action]
            visited_word.add(sel_word)
            candidate_vec = env.get_candidate_vec(sel_word)
            if sel_word == env.goal_word:
                reward, done = 1, 1
                new_state, _, _ = env.step(sel_word, current_state)
                
                agent.update_replay_memory((current_state, candidate_vec, reward, done, subspace, visited_word.copy(), step))
                agent.train(done)
                episode_reward += reward
                # print(sel_word)
                break

            new_state, reward, done = env.step(sel_word, current_state)

            if reward is None:
                continue
            
            # rew = [1  if itm>2 else 0 for itm in reward]
            reward = np.mean(reward)

            # print(sel_word, rew, reward)
            # print(new_state)

            if episodes%50==1:
                torch.save(agent.model.state_dict(), "wordle-densev4.7.pt")
            
            episode_reward += reward
              
            agent.update_replay_memory((current_state, candidate_vec, reward, done, subspace, visited_word.copy(), step))
            agent.train(done)

            current_state = new_state.copy()
            
            step += 1
            if step==7:
                break
        # break
        # print('--'*4)
        writer.add_scalar("Reward", episode_reward/step, episodes)
        writer.add_scalar("Steps Taken", step, episodes)
        writer.add_scalar("Epsilon", decaying_epsilon, episodes)
        ep_rewards.append(episode_reward)







    