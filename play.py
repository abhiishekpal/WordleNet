
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
EPISODES = 100000


# Define helper functions that are to be used
def fuse_array(arr1, arr2):
    arr1 = torch.tensor(arr1)
    arr2 = torch.tensor(arr2)
    arr1 = torch.flatten(arr1, start_dim=1)
    arr2 = torch.flatten(arr2, start_dim=1)
    arr = torch.concat([arr1, arr2], dim=1)
    return arr

def get_candidate_vec(aword):

    new_state  = np.zeros((1, 26, 5))
    for j, w in enumerate(aword):
        new_state[0][(ord(w)-97)][j] = 1

    return new_state


class Agent:

    """
    Agene that interacts with the game, suggests actions and updates its understanding
    """

    def __init__(self) -> None:
        
        self.model = DQNet()
        self.target_model = copy.deepcopy(self.model)
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-3)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.target_model = self.target_model.to(self.device)
        self.target_update_counter = 0

    def update_replay_memory(self, transition):
        
        self.replay_memory.append(transition)


    def train(self, terminal_state):
        
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # generate a sample of length MINIBATCH_SIZE from our queue of inputs for training on
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # each transition in minibatch contains: (current_state, selected_vec, new_state), reward, done, subspace, visited_word.copy(), step)
     
        self.model.eval()
        with torch.inference_mode():

            future_qs_list = []
            for transition in minibatch:
                temp = []
                for all_words in transition[3]:
                    if all_words in transition[4]:
                            continue
                    new_state  = np.zeros(transition[0][0].shape)
                    for j, w in enumerate(all_words):
                        new_state[0][(ord(w)-97)][j] = 1
                    temp.append(new_state)

                new_current_states = torch.concat([fuse_array(transition[0][2], item) for item in temp], dim=0).type(torch.float).to(self.device)
                future_qs_list.append(self.target_model(new_current_states))
            
        future_qs_list = [item.cpu().numpy() for item in future_qs_list]
        X, y = [], []
        for index, (state_tuple, reward, done, cs, vw, st) in enumerate(minibatch):
            if not done and st<6:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = np.array(reward, dtype=np.float32)

            current_qs = new_q.reshape((1))

            X.append(fuse_array(state_tuple[0], state_tuple[1]))
            y.append(torch.tensor(current_qs))


        # train over the newly created input features ad output
        X = torch.concat([itm for itm in X], dim=0).type(torch.float32).to(self.device)
        y = torch.concat([itm for itm in y], dim=0).type(torch.float32).to(self.device)
        self.model.train()
        output = self.model(X)
        loss = self.loss_fn(output.squeeze(), y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if terminal_state:
            self.target_update_counter += 1
        
        # update our target model (policy model) to the trained model
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            torch.save(self.model.state_dict(), 'model2.pth')
            self.target_model = copy.deepcopy(self.model)
            self.target_update_counter = 0

    def get_qs(self, state):
        self.model.eval()
        with torch.inference_mode():
            state = state.type(torch.float).to(self.device)
            state = state.unsqueeze(dim=0)
            return self.model(state)


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
    agent = Agent()

    for episodes in tqdm.tqdm(range(EPISODES)):

        # tracking rewards for each game 
        episode_reward = 0

        # Reset the board state and empty the visited word list
        current_state = env.reset()
        visited_word = set()
        

        # select a random candidate space from the large space of words for each game
        cand_space = list(env.CANDIDATE_SPACE)

        # fix a decaying epsilon for each game
        epsilon, decay_period, warmup_steps = 0.01, 15000, 1000
        decaying_epsilon = linearly_decaying_epsilon(decay_period, episodes, warmup_steps, epsilon)

        # starting stepping in the game
        step = 1
        while(1):
            random.shuffle(cand_space)
            subspace = list(set(cand_space[:20]+[env.goal_word]).difference(visited_word))
            random.shuffle(subspace)

            # choosing the best word with some randomness added
            if np.random.random() > decaying_epsilon:
                all_states = []                
                for cand in (subspace):
                    candidate_vec = get_candidate_vec(cand)
                    fused_state = fuse_array(current_state, candidate_vec)
                    all_states.append(fused_state)
                final_fused_state = torch.concat(all_states, dim=0)
                scores = agent.get_qs(final_fused_state).squeeze()
                action = torch.argmax(scores, dim=0)
            else:
                action = np.random.randint(0, len(subspace))
            
            # create the vector for the chosen word
            current_selected_word = list(subspace)[action]
            visited_word.add(current_selected_word) 
            candidate_vec = get_candidate_vec(current_selected_word)

            # if the chosen word is infact the goal word we update the memory with the states and reward, train a step and terminate
            if current_selected_word == env.goal_word:
                new_state, _ = env.step(current_selected_word, current_state)
                agent.update_replay_memory(((current_state, candidate_vec, new_state), 1, 1, subspace, visited_word.copy(), step))
                agent.train(1)
                episode_reward = 1
                break

            new_state, reward = env.step(current_selected_word, current_state)
            
            done = 1 if step==6 else 0
            reward = np.mean(reward)
            episode_reward = reward
            agent.update_replay_memory(((current_state, candidate_vec, new_state), reward, done, subspace, visited_word.copy(), step))
            agent.train(done)
            current_state = new_state.copy()
            
            # break if we are exceeing 6 steps (which is also wordle rule)
            if step==6:
                break
            step += 1
            if episodes%50==1:
                torch.save(agent.model.state_dict(), "wordle-dense.v6.pt")

        writer.add_scalar("Reward", episode_reward, episodes)
        writer.add_scalar("Steps Taken", step, episodes)
        writer.add_scalar("Epsilon", decaying_epsilon, episodes)








    