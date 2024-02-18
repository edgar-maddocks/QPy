import torch
from torch import no_grad
from torch.nn.functional import one_hot
from torch import nn
from torch.optim import Adam
from collections import deque
import random
import pandas as pd
import numpy as np
import warnings
import wandb

warnings.simplefilter(action='ignore', category=FutureWarning)

class DQNAgent:
    def __init__(self, env, model, epsilon = 1, min_epsilon = 0.1, epsilon_decay = 0.99, gamma = 0.5, 
                 max_mem_length = 1024, lr = 0.001, device = "cpu"):
        self.main_model = model.to(device)
        self.target_model = model.to(device)
        self.main_model.opt = Adam(params = self.main_model.parameters(),lr = lr)
        self.env = env
        self.device = device

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.max_mem_length = max_mem_length
        self.replay = ReplayBuffer(max_mem_length=max_mem_length)
        self.lr = lr

    def get_actions(self, obs):
        if random.uniform(0, 1) > self.epsilon:
            return self.env.action_space.sample()
        else:
            q_vals = self.main_model(torch.Tensor(obs).to(self.device))
            return q_vals.max(-1)[1]
        
    def update_weights(self, batch_size):
        replay_batch = self.replay.sample(batch_size)
        states = torch.stack(([torch.Tensor([x[0]]) for x in replay_batch])).to(self.device)
        rewards = torch.stack(([torch.Tensor([x[1]]) for x in replay_batch])).to(self.device)
        done_mask = torch.stack(([torch.Tensor([1]) if x[2] else torch.Tensor([0]) for x in replay_batch])).to(self.device)
        actions = [x[3] for x in replay_batch]
        next_states = torch.stack(([torch.Tensor([x[4]]) for x in replay_batch])).to(self.device)

        with no_grad():
            next_qvals = self.target_model(next_states).max(-1)[0]

        qvals = self.main_model(states)
        oh_actions = one_hot(torch.LongTensor(actions), self.env.action_space.n).to(self.device)

        loss = ((rewards + (self.gamma * (done_mask * next_qvals)) - torch.sum(qvals*oh_actions, -1))**2).mean()
        self.main_model.opt.zero_grad()
        loss.backward()
        self.main_model.opt.step()

        return loss
        
    def train(self, episodes=1000, epochs=100, steps_before_train = 1000, min_mem_size=1000, batch_size = 64, 
                update_target_interval=1000, return_data = False, use_wandb = False):
        if use_wandb:
            wandb.login()
            log = wandb.init(
                project = "DQN",
                config = {
                    "epsilon" : self.epsilon
                }
            )
        if return_data:
            loss_data = []
            reward_data = []
            epsilon_data = []
        steps_last_train = 0
        steps = 0
        next_state = self.env.reset()
        for episode in range(episodes):
            if return_data or use_wandb:
                running_reward = []
                running_loss = []
            done = False
            while not done:
                steps += 1
                steps_last_train += 1

                state = next_state.copy()
                action = self.get_actions(next_state)
                next_state, reward, done, action_taken = self.env.step(action)
                
                if return_data or use_wandb:
                    running_reward.append(reward)

                self.replay.add((state, reward, done, action_taken, next_state))

                if done:
                    next_state = self.env.reset()
                
                if len(self.replay.memory) > min_mem_size and steps_before_train < steps_last_train:
                    for epoch in range(epochs):
                        loss = self.update_weights(batch_size)
                        if return_data or use_wandb:
                            running_loss.append(loss.detach().item())               
                    print(f"Episode : {episode}, Steps taken : {steps}, Loss: {loss.detach().item()}, Reward: {sum(running_reward)}")
                    steps_last_train = 0
                
                if steps % update_target_interval == 0:
                    self.update_target_model()
                    print(f"Target Model Updated at episode: {episode}")

            self.update_epsilon()
            if use_wandb:
                if len(running_loss) > 0:
                    wandb.log({
                        "loss" : max(running_loss),
                        "reward" : sum(running_reward),
                        "epsilon" : self.epsilon
                    })
                else:
                    wandb.log({
                        "reward" : sum(running_reward),
                        "epsilon" : self.epsilon
                    })
            if return_data:
                if len(running_loss) > 0:
                    loss_data.append(max(running_loss))
                reward_data.append(sum(running_reward))
                epsilon_data.append(self.epsilon)
                
        
        return loss_data, reward_data, epsilon_data


    def update_target_model(self):
        self.target_model.load_state_dict(self.main_model.state_dict())

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
    
class Model(nn.Module):
    def __init__(self, obs_shape, action_shape, neurons = 256, device = "cpu"):
        super(Model, self).__init__()

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(self.obs_shape[1], neurons),
            nn.ReLU(),
            nn.Linear(neurons, self.action_shape),
        ).to(device)

    def forward(self, input):
        input = torch.Tensor(input).to(self.device)
        return self.net(input)


class ReplayBuffer:
    def __init__(self, max_mem_length = 1024):
        self.max_mem_length = max_mem_length
        self.memory = deque(maxlen = self.max_mem_length)

    def add(self, info):
        self.memory.append(info)

    def sample(self, batch_size):
        assert batch_size < len(self.memory)
        return random.sample(self.memory, batch_size)



