
import torch
from wordle_simulator import Wordle
from play import Agent, fuse_array, get_candidate_vec
import numpy as np

import torch
import tqdm
import random
import json
 

env = Wordle()

agent = Agent()
agent.model.load_state_dict(torch.load('wordle-dense.v6.pt'))


total_solved = 0
steps_transitioned = {}
for iter in range(100):  
    done = False
    step = 1
    current_state = env.reset()
    visited_word = set()
    print(env.goal_word)
    print('----')
    steps_transitioned[iter] = {"goal": env.goal_word, "steps": []}
    candidates = list(env.CANDIDATE_SPACE)
    random.shuffle(candidates)
    subspace = candidates[:200] 

    
    while not done:

        scores = []
        subspace = list(set(subspace).difference(visited_word))
        random.shuffle(subspace)
        subspace2 = list(set(subspace + [env.goal_word]))
        random.shuffle(subspace2)

        all_states = []                
        for cand in (subspace2):
            candidate_vec = get_candidate_vec(cand)
            fused_state = fuse_array(current_state, candidate_vec)
            all_states.append(fused_state)
        final_fused_state = torch.concat(all_states, dim=0)
        scores = agent.get_qs(final_fused_state).squeeze()
        action = torch.argmax(scores, dim=0)

        current_selected_word = list(subspace2)[action]
        visited_word.add(current_selected_word)

        print(current_selected_word, step)
        new_state, _ = env.step(current_selected_word, current_state)
        current_state = new_state.copy()
        steps_transitioned[iter]["steps"].append(current_selected_word)
        if current_selected_word==env.goal_word:
            total_solved += 1
            break
        step += 1
        if step==7:
            break
print(f"Solved_percentage: {total_solved/len(steps_transitioned)*100}")
with open("results.json", "w") as fp:
    json.dump(steps_transitioned, fp)

  