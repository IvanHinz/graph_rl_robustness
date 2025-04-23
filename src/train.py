import random
import torch
import torch.optim as optim
import torch.nn as nn
from collections import deque
import networkx as nx
from src.graph_env import GraphEnv
from src.utils import perturb_graph

from src.adv_attacks import fgsm_attack, eaan_attack, eacn_attack #, fgsm_actor, fgsm_critic
from src.models import DQN, ActorCritic

# masked because we cannot go to all edges from some defined vertice
def train_dqn_masked(env, num_episodes=1000, gamma=0.99, use_fgsm=False, domain_randomization=False, train=True):
    num_nodes = env.num_nodes
    online_net = DQN(num_nodes, hidden_size=num_nodes) # online network
    target_net = DQN(num_nodes, hidden_size=num_nodes) # target network
    target_net.load_state_dict(online_net.state_dict())
    optimizer = optim.Adam(online_net.parameters(), lr=1e-3)
    # define the loss
    loss_fn = nn.MSELoss()
    buffer = deque(maxlen=10000)

    eps = 1.0
    eps_min = 0.1
    eps_decay = 0.995
    batch_size = 64
    rewards_log = []
    optimal_misses = []
    # -shortest path according to dijkstra + 100 (to make it compatible with our rewards function)
    shortest = -nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight') + 100

    was_optimal = False # added on my thought way
    founded_ep = num_episodes
    domain_updated = num_episodes // 3
    for episode in range(num_episodes):
        # if needed we use domain_randomization
        if domain_randomization and train and episode % domain_updated == 0:
            g = env.our_graph.copy()
            g_pert = perturb_graph(g)
            env=GraphEnv(g_pert)
            shortest = -nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight') + 100
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        total_r = 0
        done = False

        while not done:
            if use_fgsm:
                state_tensor = fgsm_attack(state_tensor.clone().detach(), online_net, epsilon=0.1)

            valid = env.get_valid_actions()
            # if was_optimal:
            #     with torch.no_grad():
            #         q_values = online_net(state_tensor.unsqueeze(0)).squeeze(0)
            #         q_values_masked = q_values.clone()
            #         invalid = list(set(range(num_nodes)) - set(valid))
            #         q_values_masked[invalid] = -float('inf')
            #         action = int(torch.argmax(q_values_masked))
            # else:
            if random.random() < eps:
                action = random.choice(valid)
            else:
                with torch.no_grad():
                    q_values = online_net(state_tensor.unsqueeze(0)).squeeze(0)
                    q_values_masked = q_values.clone()
                    invalid = list(set(range(num_nodes)) - set(valid))
                    q_values_masked[invalid] = -float(10**9)
                    action = int(torch.argmax(q_values_masked))

            next_state, reward, done, _ = env.step(action)
            next_tensor = torch.tensor(next_state, dtype=torch.float32)
            buffer.append((state_tensor, action, reward, next_tensor, done))
            state_tensor = next_tensor
            total_r += reward

            if len(buffer) >= batch_size:
                batch = random.sample(buffer, batch_size)
                st, act, rew, nst, d = zip(*batch)
                st = torch.stack(st)
                act = torch.tensor(act)
                rew = torch.tensor(rew, dtype=torch.float32)
                nst = torch.stack(nst)
                d = torch.tensor(d, dtype=torch.bool)
                q_predicted = online_net(st).gather(1, act.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target_net(nst).max(1)[0]
                    target_q = rew + gamma * q_next * (~d)
                loss = loss_fn(q_predicted, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards_log.append(total_r)
        optimal_misses.append(int(abs(total_r - shortest) > 1e-3))
        if optimal_misses[-1] == 0 and founded_ep==num_episodes:
            # was_optimal = True
            founded_ep = episode
        if eps > eps_min:
            eps *= eps_decay
        if (episode+1) % 50 == 0:
            if use_fgsm:
                print(f"DQN attacked with FGSM on episode {episode + 1}, reward={total_r:.2f}, miss_optimal={optimal_misses[-1]}")
            else:
                print(f"DQN without an attack on episode {episode + 1}, reward={total_r:.2f}, miss_optimal={optimal_misses[-1]}")
        if (episode+1) % 100 == 0:
            target_net.load_state_dict(online_net.state_dict())

    return online_net, rewards_log, optimal_misses, founded_ep

def train_actor_critic(env, num_episodes=1000, gamma=0.99, attack=None, epsilon=0.1, domain_randomization=False, train=True):
    num_nodes = env.num_nodes
    # edge_index = make_edge_index(env.adjacency_matrix)  #
    model = ActorCritic(num_nodes, hidden_size=num_nodes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    rewards_log = []
    optimal_misses = []
    shortest = -nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight') + 100

    founded_ep = num_episodes
    domain_updated = num_episodes//5
    for episode in range(num_episodes):
        # if needed we use domain randomization
        if domain_randomization and train and episode % domain_updated == 0:
            g = env.our_graph.copy()
            g_pert = perturb_graph(g)
            env = GraphEnv(g_pert)
            shortest = -nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight') + 100
        state = env.reset()
        state_oh = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        log_probs = []
        values = []
        rewards = []
        masks = []

        while not done:
            if attack == 'eaan':
                state_oh = eaan_attack(state_oh.clone(), model, epsilon)
            elif attack == 'eacn':
                state_oh = eacn_attack(state_oh.clone(), model, epsilon)
            # elif attack == 'fgsm_critic':
            #     state_oh = fgsm_critic(state_oh.clone(), model, epsilon)
            # elif attack == 'fgsm_actor':
            #     state_oh = fgsm_actor(state_oh.clone(), model, epsilon)

            logits, value = model(state_oh.unsqueeze(0))
            valid = env.get_valid_actions()
            logits = logits.squeeze(0)
            mask = torch.full((num_nodes,), float(-10**9))
            mask[valid] = 0.0
            # print(logits)
            masked_logits = logits + mask
            probs = torch.softmax(masked_logits, dim=-1)
            # print(probs)
            dist = torch.distributions.Categorical(probs)
            # print(dist)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            state_oh = torch.tensor(next_state, dtype=torch.float32)
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            masks.append(torch.tensor([1 - done], dtype=torch.float32))
            total_reward += reward

        rewards_log.append(total_reward)
        optimal_misses.append(int(abs(total_reward - shortest) > 1e-3))
        returns = []
        G = torch.tensor([0.0])
        for r, m in zip(reversed(rewards), reversed(masks)):
            G = r + gamma * G * m
            returns.insert(0, G)

        returns = torch.cat(returns)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if optimal_misses[-1] == 0 and founded_ep == num_episodes:
            founded_ep = episode
        if (episode+1) % 50 == 0:
            if attack:
                print(f"A2C attacked with {attack} on episode {episode + 1}, reward={total_reward:.2f}, miss_optimal={optimal_misses[-1]}")
            else:
                print( f"A2C without an attack on episode {episode + 1}, reward={total_reward:.2f}, miss_optimal={optimal_misses[-1]}")

    return model, rewards_log, optimal_misses, founded_ep

