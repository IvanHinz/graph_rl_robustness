import numpy as np
import random

def moving_average(data, window_size=50):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        # extract the data for the next window size elements
        window = data[i: i + window_size]
        # compute the average
        window_average = np.mean(window)
        moving_averages.append(window_average)

    # get list of the values of moving average
    return np.array(moving_averages)

def perturb_graph(graph, weight_change=5, num_changes=20):
    perturb_g = graph.copy()
    edges = list(perturb_g.edges())
    for _ in range(num_changes):
        u, v = random.choice(edges)
        # we randomly change the defined number of edges (num_changes)
        # the strength of change is defined by weight_change
        perturb_g[u][v]['weight'] = max(1, perturb_g[u][v]['weight'] + random.choice([-weight_change, weight_change]))
    return perturb_g