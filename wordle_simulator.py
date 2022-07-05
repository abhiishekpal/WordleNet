import random
import numpy as np
from nltk.corpus import words


class Wordle:

    def __init__(self, size=(6, 5)) -> None: 

        self.size = size
        word_list = words.words()
        self.all_words = [wl.lower() for wl in word_list if len(wl)==size[1]]
        self.goal_word = random.choices(population=self.all_words, k=1)[0]
        self.visited_word = set()
        # print('Goal: ', self.goal_word)
        self.ACTION_SPACE_SIZE = len(self.all_words)
        self.box = [['']*size[1]]*size[0]
        self.scores = [[-1]*size[1]]*size[0]
        self.word_list = []
        self.reward_list = []
        self.word_count = {i: 0 for i in range(26)}
        for i in range(self.size[1]):
            self.word_count[ord(self.goal_word[i])-97] += 1
    
    def action(self, idx):
        word = self.all_words[idx]
    
    
        if len(word) != self.size[1] or word not in self.all_words or word in self.visited_word:
            return []
        # print('\t', word)
        self.visited_word.add(word)
        reward = [-1] * self.size[1]
        wc = self.word_count.copy()
        for i in range(self.size[1]):
            if word[i] == self.goal_word[i]:
                reward[i] = 1
                wc[ord(word[i])-97] -= 1
        
        for i in range(self.size[1]):
            if reward[i]==-1 and word[i] in self.goal_word and wc[ord(word[i])-97]:
                reward[i] = 0
                wc[ord(word[i])-97] -= 1

        self.reward_list.append(reward)
        self.word_list.append(word)

        return reward

    def show_board(self):
        for wl in self.word_list:
            print(wl)
        print('-----\n'*2)
        for rl in self.reward_list:
            print(rl)
        print('-----\n'*4)

    def reset(self):
        current_state  = np.zeros(tuple([2])+self.size)
        self.visited_word = set()
        return current_state

    def step(self, aword):
        
        l2 = len(self.reward_list)
        reward = self.action(aword)

        if len(self.reward_list)==l2:
            return None, None, None

        new_state  = np.zeros(tuple([2])+self.size)

        for i, word in enumerate(self.word_list):
            for j, w in enumerate(word):
                new_state[0][i][j] = ord(w)
        for i, reward in enumerate(self.reward_list):
            for j, r in enumerate(reward):
                new_state[1][i][j] = r

        return new_state, reward, len(self.word_list)==self.size[0]
        
        

        



    

    

    
               
        
    

