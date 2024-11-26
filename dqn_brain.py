import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        _, h = self.gru(x)
        return h.squeeze(0)


class DQN(nn.Module):
    def __init__(self, k, hidden_size=3):
        super(DQN, self).__init__()
        self.k = k
        self.hidden_size = hidden_size
        
        self.gru = GRUModel(input_size=3, hidden_size=hidden_size)
        
        self.fc1 = nn.Linear(5 * k + 2 + hidden_size, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, k)  

    def forward(self, candidate_state, solution_state, global_state):
        
        sequence = solution_state["sequence"]  
        gru_embedding = self.gru(sequence)
        
        solution_features = torch.cat([
            gru_embedding,
            solution_state["size"].unsqueeze(1),
            solution_state["sum_items"].unsqueeze(1),
            solution_state["coverage"].unsqueeze(1)
        ], dim=1)
        state_features = torch.cat([
            candidate_state.view(candidate_state.size(0), -1),
            solution_features,
            global_state
        ], dim=1)
        
        x = torch.relu(self.fc1(state_features))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values



class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
        

class DQNAgent:
    def __init__(self, k, lr, gamma, epsilon, epsilon_decay, epsilon_min, buffer_capacity, batch_size):
        self.k = k
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        self.q_net = DQN(k).to(device)
        self.target_net = DQN(k).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.k - 1)
        else:
            candidate_state, solution_state, global_state = state
            candidate_state = torch.FloatTensor(candidate_state).unsqueeze(0).to(device)
            solution_state = {
                key: torch.FloatTensor(value).unsqueeze(0).to(device)
                if isinstance(value, np.ndarray) else value for key, value in solution_state.items()
            }
            global_state = torch.FloatTensor(global_state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_net(candidate_state, solution_state, global_state)
            return torch.argmax(q_values).item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        candidate_states, solution_states, global_states = zip(*states)
        candidate_states = torch.FloatTensor(candidate_states).to(device)
        solution_states = {
            key: torch.FloatTensor([s[key] for s in solution_states]).to(device)
            if key != "sequence" else torch.FloatTensor(np.stack([s[key] for s in solution_states])).to(device)
            for key in solution_states[0]
        }
        global_states = torch.FloatTensor(global_states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)
        next_candidate_states, next_solution_states, next_global_states = zip(*next_states)
        
        q_values = self.q_net(candidate_states, solution_states, global_states).gather(1, actions)
        
        with torch.no_grad():
            next_candidate_states = torch.FloatTensor(next_candidate_states).to(device)
            next_solution_states = {
                key: torch.FloatTensor([s[key] for s in next_solution_states]).to(device)
                if key != "sequence" else torch.FloatTensor(np.stack([s[key] for s in next_solution_states])).to(device)
                for key in next_solution_states[0]
            }
            next_global_states = torch.FloatTensor(next_global_states).to(device)
            next_q_values = self.target_net(next_candidate_states, next_solution_states, next_global_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.criterion(q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def train_dqn(env, num_episodes, k, target_update_freq, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, buffer_capacity=10000, batch_size=64):
    agent = DQNAgent(k, lr, gamma, epsilon, epsilon_decay, epsilon_min, buffer_capacity, batch_size)
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()  
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)  
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            agent.train()
        
        if episode % target_update_freq == 0:
            agent.update_target_network()

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.3f}")

    return rewards


def save_model(model, optimizer, file_path, epoch=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    if epoch is not None:
        checkpoint['epoch'] = epoch
    torch.save(checkpoint, file_path)
    print(f"Model saved to {file_path}")


def load_model(model, optimizer, file_path):
    checkpoint = torch.load(file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', None)
    print(f"Model loaded from {file_path}")
    return epoch



def create_agent_from_checkpoint(file_path, k, lr, gamma, epsilon, epsilon_decay, epsilon_min, buffer_capacity,
                                 batch_size):

    model = DQN(k).to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    load_model(model, optimizer, file_path)

    agent = DQNAgent(
        k=k,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size
    )
    
    agent.q_net = model
    agent.optimizer = optimizer
    return agent
