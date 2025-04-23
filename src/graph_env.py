import random
import numpy as np
import networkx as nx

def create_graph(num_nodes, num_edges):
    while True:
        g = nx.gnm_random_graph(num_nodes, num_edges)
        if nx.is_connected(g):
            break
    for (u, v) in g.edges():
        g[u][v]['weight'] = random.randint(1, 10)
    return g

class GraphEnv:
    def __init__(self, our_graph, max_steps=100):
        self.num_steps_taken = 0
        self.current_node = None
        self.our_graph = our_graph
        self.num_nodes = our_graph.number_of_nodes()
        # define the starting node
        self.start_node = 0
        # define the goal node
        self.goal_node = self.num_nodes - 1
        # define the max_steps that we can go from node to node
        self.max_steps = max_steps

        # adjacency matrix
        self.adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        self.weights = np.full((self.num_nodes, self.num_nodes), -1.0, dtype=float)

        # put the values in weight matrix and adjacency matrix
        for (u, v) in self.our_graph.edges():
            w = self.our_graph[u][v]['weight']
            self.adjacency_matrix[u,v], self.adjacency_matrix[v,u] = 1, 1
            self.weights[u,v], self.weights[v,u] = w, w

        self.reset()

    def reset(self):
        self.current_node = self.start_node
        self.num_steps_taken = 0
        return self._get_obs()

    def _get_obs(self):
        # returns the list of weights to adjacent node and for the node itself
        return self.weights[self.current_node].copy()

    def get_valid_actions(self):
        # get the valid nodes to go from current node
        return np.where(self.adjacency_matrix[self.current_node] == 1)[0]

    def step(self, action):
        self.num_steps_taken += 1
        done = False
        info = {}
        # if self.adjacency_matrix[self.current_node, action] == 0 or action == self.current_node:
        #     raise ValueError("invalid action")
        # reward from the current node to the next node (defined as action)
        cost = self.weights[self.current_node, action]
        reward = -float(cost)

        shaped_reward = -float(cost) # 0.5 * (old_dist - new_dist)
        # current node becomes the node where we resulted to be
        self.current_node = action
        # node where we are is valid
        info["invalid"] = False
        if self.current_node == self.goal_node:
            # if we reached the target node
            shaped_reward += 100.0
            done = True
        if self.num_steps_taken >= self.max_steps:
            done = True
        # if we made too much steps we also assume that the done is True
        # return observation, reward, information if done, and info
        return self._get_obs(), shaped_reward, done, info

