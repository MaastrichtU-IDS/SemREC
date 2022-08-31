import numpy as np
import random

class Env():
    def __init__(self,triples,num_actions=1):

        self.triples = triples
        self.done    = False
        self.action_space_n = num_actions
    
    def step(self,s,r):
        ''' s -> r -> t '''
        reward = -0.1
        next_state = s
        
        for triplet in self.triples:
            if triplet[0] == s and triplet[1] == r:
                # good state
                next_state = triplet[2]
                if s == next_state: # don't encourage self loop
                    reward = 0 # -0.1
                else:
                    reward = 0.1
            
                if next_state == self.target:
                    reward = 1
                    self.done = True    
                    break
                
        return next_state,reward, self.done
                
            
    def reset(self):
        self.done = False
        triplet = random.choice(self.triples)
        self.source = triplet[0]
        self.relation = triplet[1]
        self.target = triplet[2]
        return self.source
                   
    def reward(self):
        pass
    
    
    
    