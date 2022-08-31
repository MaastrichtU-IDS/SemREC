import torch
from torch import nn
import copy
from collections import deque
from torch.distributions import Bernoulli
import numpy as np
import random

def process_agent_samples(train_quads,new_samples):
    uniques=0
    nounique=0
    unique_agent_samples = []
    for new_sample in new_samples:
        if len(new_sample)==6 and new_sample[-1]>1:
            found_sample=False
            for quad in train_quads:
                if len(quad) == len(new_sample[:-1]):
                    count=0
                    for j in range(len(quad)):
                        if quad[j] == new_sample[j]:
                            count+=1
                    if count == len(quad):
                        found_sample=True
                        break
            if found_sample:
                #print(new_sample,' not unique')
                nounique+=1
            else:
                #print(new_sample,' is unique')
                uniques+=1
                unique_agent_samples.append((new_sample[0],
                                       new_sample[1],
                                       new_sample[2],
                                       new_sample[3],
                                       new_sample[4]))
    print('unique ratio: {:0.02f}'.format(uniques/(uniques+nounique)))        
    return unique_agent_samples

class ExperienceReplay(object):
      def __init__(self, length):
        self.len = length
        self.experience_replay = deque(maxlen=length)

      def collect(self, experience):
        self.experience_replay.append(experience)
        return

      def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)
        sample  = random.sample(self.experience_replay, sample_size)
        state   = torch.tensor([exp[0] for exp in sample]).float()
        action  = torch.tensor([exp[1] for exp in sample]).float()
        reward  = torch.tensor([exp[2] for exp in sample]).float()
        next_state = torch.tensor([exp[3] for exp in sample]).float()
        return state, action, reward, next_state

class DQN_Network:

    def __init__(self, layer_size_list, lr, seed=1423,exp_replay_size = 20):
        torch.manual_seed(seed)
        self.policy_net = self.create_network(layer_size_list)
        self.target_net = copy.deepcopy(self.policy_net)
        
        self.memory = ExperienceReplay(exp_replay_size)
  
        self.loss_fn = torch.nn.MSELoss() # the loss function
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.step = 0
        self.gamma = torch.tensor(0.95).float()
        return

    def create_network(self, layer_size_list):
        assert len(layer_size_list) > 1

        layers = []
        for i in range(len(layer_size_list) - 1):
            linear = nn.Linear(layer_size_list[i], layer_size_list[i + 1])

            if i < len(layer_size_list) - 2:
              activation = nn.Tanh()
            else:
              activation = nn.Identity()

            layers += (linear, activation)
        return nn.Sequential(*layers)

    def load_pretrained_model(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path))

    def save_trained_model(self, model_path="cartpole-dqn.pth"):
        torch.save(self.policy_net.state_dict(), model_path)
        
    def get_action(self, state, action_space_len, epsilon):
   
        with torch.no_grad():
            Qp = self.policy_net(torch.from_numpy(state).float())

        if np.random.rand() < epsilon:
          a = torch.tensor(random.choice(list(range(action_space_len))))    
        else:
          a = torch.argmax(Qp)
      
        return a
    
    def _supervised_train(self, node2vec,rel2vec,batch_size):
        state, action, reward, next_state = self.memory.sample_from_experience(sample_size=batch_size)

        # predict expected return of current state using main network    
        state_embs = torch.index_select(torch.tensor(node2vec),0,torch.tensor([int(x) for x in state]))
        acts_embs  = torch.index_select(torch.tensor(rel2vec),0,torch.tensor([int(x) for x in action]))
        Qs = self.policy_net(torch.cat((state_embs,acts_embs),dim=1))
        
        # get target return using target network
        next_state_embs = torch.index_select(torch.tensor(node2vec),0,torch.tensor([int(x) for x in next_state]))
        next_act_embs   = torch.zeros_like(next_state_embs) # ???? acts_embs + 0_vec ------------------> fix
        Qmax = self.target_net(torch.cat((next_state_embs,next_act_embs),dim=1)).max(dim=1).values
        Qt = torch.zeros_like(Qs)
        for i in range(len(Qmax)):
            Qt[i,:] = Qs[i,:]
            Qt[i,action[i].long()] = Qmax[i]* self.gamma + reward[i]

        # compute the loss
        loss = self.loss_fn( Qs ,Qt)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.step += 1
        if self.step % 5 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
    
    def train(self,env,node2vec,rel2vec,episodes = 1000,eps_decay_rate= 0.999):
        new_samples = []
        
        losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
        
        # initiliaze experiance replay
        index = 0       
        for i in range(self.memory.len):
            obs = env.reset()
            done = False
            while not done:
                state = np.concatenate( [node2vec[obs,:],np.zeros_like(rel2vec[0,:])] )
                A = self.get_action( state, env.action_space_n, epsilon=1)
                obs_next, reward, done = env.step(obs,A.item())
                self.memory.collect([obs, A.item(), reward, obs_next])
                obs = obs_next
                index += 1
                if index > self.memory.len:
                    break
                    
        epsilon = 1 
        for i in range(episodes):
            episod_transitions = []
            obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
            episod_transitions+=[obs]
                
            while not done:
                ep_len += 1
                state = np.concatenate( [node2vec[obs,:],np.zeros_like(rel2vec[0,:])] )
                A = self.get_action( state, env.action_space_n, epsilon)
                obs_next, reward, done = env.step(obs,A.item())
                episod_transitions+=[A.item()]
                episod_transitions+=[obs_next]
                self.memory.collect([obs, A.item(), reward, obs_next])

                obs = obs_next
                rew += reward
                index += 1
                
                
                if  ep_len > 5 or done:
                    for j in range(4):
                        loss = self._supervised_train(node2vec,rel2vec, batch_size=16)
                        losses += loss
                    break

            # epsilon decay rule
            epsilon = max(epsilon * eps_decay_rate, 0.01)
            
            if done == True and ep_len>1:
                new_samples.append(episod_transitions+[rew])
            
            if i % (episodes//10) == 0:
                print('epoch {}\tep_len {}\taverage loss {:0.02f}\treward {:0.02f}\tdone {}\teps {:0.02f}'.format(i,ep_len,losses,rew,done,epsilon))

        losses_list.append(losses / ep_len), reward_list.append(rew)
        episode_len_list.append(ep_len), epsilon_list.append(epsilon)
        
        
        return new_samples