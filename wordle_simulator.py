import random
import numpy as np
from nltk.corpus import words


class Wordle:
    ALL_WORDS_OG = []
    with open('valid-wordle-words.txt', 'r') as fp:
        for line in fp:
            ALL_WORDS_OG.append(line.replace('\n', '').strip())
    def __init__(self, size=(6, 5)) -> None: 
        
        # select all words of length size[1] from the vocabulary present in above text file
        self.size = size
        self.all_words = [wl.lower() for wl in Wordle.ALL_WORDS_OG if len(wl)==size[1]]
        self.CANDIDATE_SPACE = set(self.all_words)
        self.goal_word = random.choices(population=self.all_words, k=1)[0]


        self.visited_word = set()

        # creating a counter for each word, this will be utilized in computing the return state
        self.reward_list = []
        self.word_count = {i: 0 for i in range(26)}
        for i in range(self.size[1]):
            self.word_count[ord(self.goal_word[i])-97] += 1
    
    def action(self, word):

        if len(word) != self.size[1] or word not in self.all_words or word in self.visited_word:
            return None
            
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

    def reset(self):
        """
        Resetting the state of our board
        """
        self.goal_word = random.choices(population=self.all_words, k=1)[0]
        self.visited_word = set()
        # creating a counter for each word, this will be utilized in computing the return state
        self.reward_list = []
        self.word_count = {i: 0 for i in range(26)}
        for i in range(self.size[1]):
            self.word_count[ord(self.goal_word[i])-97] += 1

        current_state  = np.zeros((1, 26, 5))
        self.visited_word = set()
        return current_state

    def step(self, aword, prev_state):
        
        reward = self.action(aword)

        if reward is None:
            return None, None
        
        new_state  = prev_state.copy() 
        for j, w in enumerate(aword):
            if new_state[0][(ord(w)-97)][j]==0:
                 new_state[0][(ord(w)-97)][j] = reward[j]
            else:
                new_state[0][(ord(w)-97)][j] = max(prev_state[0][ord(w)-97][j], reward[j])
       
       
        return new_state, reward
        
        

        



    

    

    
               
        
    

