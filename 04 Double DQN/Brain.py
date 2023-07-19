
from Hippocampus import Memory
from PFC import DeepQNetwork
import torch as torch
import numpy as np


class Agent:
    def __init__(self, gamma, epsilon_initial, epsilon_decay, epsilon_final, memory_size, input_dimensions,
                 action_dimensions, learning_rate, weight_decay, batch_size, replace, algorithm, environment,
                 directory):
        self.gam = gamma
        self.eps_i = epsilon_initial
        self.eps_d = epsilon_decay
        self.eps_f = epsilon_final
        self.mem_dim = memory_size
        self.inp_dim = input_dimensions
        self.act = action_dimensions
        self.lr = learning_rate
        self.l2_reg = weight_decay
        self.bat_dim = batch_size
        self.rep = replace
        self.alg = algorithm
        self.env = environment
        self.dic = directory

        self.lrn_stp = 0
        self.act_int = [i for i in range(action_dimensions)]

        self.memory = Memory(self.mem_dim, self.inp_dim)

        self.q_eva = DeepQNetwork(network_name=self.env + '_' + self.alg + '_Double_Q_Eval_Network.pth',
                                  directory=self.dic, input_dimensions=self.inp_dim, action_dimensions=self.act,
                                  learning_rate=self.lr, weight_decay=self.l2_reg)

        self.q_tar = DeepQNetwork(network_name=self.env + '_' + self.alg + '_Double_Q_Target_Network.pth',
                                  directory=self.dic, input_dimensions=self.inp_dim, action_dimensions=self.act,
                                  learning_rate=self.lr, weight_decay=self.l2_reg)

    def decision(self, state):
        if np.random.random() > self.eps_i:
            self.q_eva.eval()

            sta = torch.tensor(state, dtype=torch.float).to(self.q_eva.dev)

            with torch.no_grad():
                q_val = self.q_eva(sta)

            action = torch.argmax(q_val).item()

            self.q_eva.train()
        else:
            action = np.random.choice(self.act_int)

        return action

    def saving(self, state, action, reward, future_state, terminal):
        self.memory.save(state, action, reward, future_state, terminal)

    def sampling(self):
        state, action, reward, future_state, terminal = self.memory.sampler(self.bat_dim)

        state = torch.tensor(state).to(self.q_eva.dev)
        action = torch.tensor(action).to(self.q_eva.dev)
        reward = torch.tensor(reward).to(self.q_eva.dev)
        future_state = torch.tensor(future_state).to(self.q_eva.dev)
        terminal = torch.tensor(terminal).to(self.q_eva.dev)

        return state, action, reward, future_state, terminal

    def target_network_update(self):

        if self.lrn_stp % self.rep == 0:
            self.q_tar.load_state_dict(self.q_eva.state_dict())

    def epsilon_update(self):

        if self.eps_i > self.eps_f:
            self.eps_i = self.eps_i - self.eps_d

        else:
            self.eps_i = self.eps_f

    def saving_model(self):
        print('... Saving Model ...')
        self.q_eva.save_checkpoint()
        self.q_tar.save_checkpoint()

    def loading_model(self):
        print('... Loading Model ...')
        self.q_eva.load_checkpoint()
        self.q_tar.load_checkpoint()

    def learn(self):

        if self.memory.ctr < self.bat_dim:
            return

        self.q_eva.opt.zero_grad()

        self.target_network_update()

        sta, act, rwd, fut_sta, ter = self.sampling()

        bat_idx = np.arange(self.bat_dim, dtype=np.int32)
        q_pre = self.q_eva(sta)[bat_idx, act]

        fut_act = torch.argmax(self.q_eva(fut_sta), dim=-1)

        self.q_tar.eval()

        with torch.no_grad():
            q_fut = self.q_tar(fut_sta)

        self.q_tar.train()

        q_fut[ter] = 0.0

        td_tar = rwd + self.gam * q_fut[bat_idx, fut_act]
        lss = self.q_eva.lss(td_tar, q_pre).to(self.q_eva.dev)
        lss.backward()
        self.q_eva.opt.step()

        self.lrn_stp += 1

        self.epsilon_update()
