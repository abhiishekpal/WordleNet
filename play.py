
from wordle_simulator import Wordle
from model import DQNet
import copy
from collections import deque
import random
import numpy as np
import tqdm
import torch.nn as nn
import torch


REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
EPISODES = 20000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:

    def __init__(self, input_dimension, output_dimension) -> None:
         
        self.model = DQNet(input_dimension, output_dimension)
        self.target_model = copy.deepcopy(self.model)
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.01)
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
        self.model.eval()
        with torch.no_grad():
            
            current_states = np.array([transition[0] for transition in minibatch] )
            current_states = torch.tensor([item for item in current_states], dtype=torch.float)
            if torch.cuda.is_available():
                current_states = current_states.cuda()
            current_qs_list = self.model(current_states)
            
            new_current_states = np.array([transition[3] for transition in minibatch])
            new_current_states = torch.tensor([item for item in new_current_states], dtype=torch.float)
            if torch.cuda.is_available():
                new_current_states = new_current_states.cuda()
            
            future_qs_list = self.target_model(new_current_states)
            
        future_qs_list = future_qs_list.cpu().numpy()
        current_qs_list = current_qs_list.cpu().numpy()

        X = []
        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            
            current_qs =  current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
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




if __name__=='__main__':
    epsilon = 0.1

    env = Wordle()
    agent = Agent(env.size, env.ACTION_SPACE_SIZE)
    ep_rewards = [-200]


    for episodes in tqdm.tqdm(range(EPISODES)):

        episode_reward = 0
        step = 1
        env = Wordle()
        current_state = env.reset()
        done = False
        iter = 0
        while not done:
            
            if np.random.random() > epsilon:
                sort_action = np.argsort(agent.get_qs(current_state).cpu().numpy())[0]
                for i in range(len(sort_action)):
                    action = sort_action[i] 
                    new_state, reward, done = env.step(action)
                    if done == None:
                        done = False
                        continue
                    else:
                        break  
                      
            else:
                action = np.random.randint(0, env.ACTION_SPACE_SIZE)
                new_state, reward, done = env.step(action)

            if done==None:
                done = False
                continue
            rew = 0
            for itm in reward:
                if itm  == -1:
                    rew -= 2
                if itm == 0:
                    rew -= 2
                else:
                    rew += 2
            reward = rew
            if episodes%500==1:
                print('Reward:', reward)
            # reward = np.sum(reward) 
            
            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done)

            current_state = new_state

            step += 1
            if step==7:
                break
        ep_rewards.append(episode_reward)







    