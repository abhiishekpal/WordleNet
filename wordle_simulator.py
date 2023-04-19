import random
import numpy as np
from nltk.corpus import words


class Wordle:
    ALL_WO = words.words()
    random.shuffle(ALL_WO)
    # ALL_WORDS_OG = [wl.lower() for wl in ALL_WO]
    ALL_WORDS_OG = []
    with open('valid-wordle-words.txt', 'r') as fp:
        for line in fp:
            ALL_WORDS_OG.append(line.replace('\n', '').strip())
    def __init__(self, size=(6, 5)) -> None: 

        self.size = size
        self.all_words = [wl.lower() for wl in Wordle.ALL_WORDS_OG if len(wl)==size[1]]
        self.CANDIDATE_SPACE = set(self.all_words)
        self.goal_word = random.choices(population=self.all_words, k=1)[0]

        # print(self.goal_word)
        self.visited_word = set()


        self.ACTION_SPACE_SIZE = len(self.all_words)

        self.word_list = []
        self.reward_list = []
        self.word_count = {i: 0 for i in range(26)}
        for i in range(self.size[1]):
            self.word_count[ord(self.goal_word[i])-97] += 1
    
    def action(self, word):

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
                reward[i] = 0.5
                wc[ord(word[i])-97] -= 1

        return reward

    def get_candidate_vec(self, aword):

        new_state  = np.zeros((1, 26, 5))
        for j, w in enumerate(aword):
            new_state[0][(ord(w)-97)][j] = 1

        return new_state

    def show_board(self):
        for wl in self.word_list:
            print(wl)
        print('-----\n'*2)
        for rl in self.reward_list:
            print(rl)
        print('-----\n'*4)

    def reset(self):
        current_state  = np.zeros((1, 26, 5))
        self.visited_word = set()
        return current_state

    def step(self, aword, prev_state):
        
        reward = self.action(aword)
        new_state  = prev_state.copy() 
        for j, w in enumerate(aword):
            if new_state[0][(ord(w)-97)][j]==0:
                 new_state[0][(ord(w)-97)][j] = reward[j]
            else:
                new_state[0][(ord(w)-97)][j] = max(prev_state[0][ord(w)-97][j], reward[j])
       
       
        return new_state, reward, len(self.word_list)==self.size[0]
        
        

        



    

    

    
               
        
    

