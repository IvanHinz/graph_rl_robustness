import torch
import networkx as nx

# evaluation of the performance of the model on perturbed graph
def evaluate_on_perturbed(model, graph, env_cls, num_episodes=100):
    env = env_cls(graph)
    num_nodes = env.num_nodes
    # -shortest according to dijkstra + 100 to be compatible with the rewards in environment
    shortest = -nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight') + 100
    rewards_log = []
    optimal_misses = []

    found_episode = num_episodes
    for episode in range(num_episodes):
        state = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)
        total_r = 0
        done = False

        while not done:
            valid = env.get_valid_actions()
            with torch.no_grad():
                  output = model(state_tensor.unsqueeze(0))
                  if isinstance(output, tuple):
                      logits = output[0].squeeze(0)
                  else:
                      logits = output.squeeze(0)
                  q_values_masked = logits.clone()
                  invalid = list(set(range(num_nodes)) - set(valid))
                  q_values_masked[invalid] = -float('inf')
                  action = int(torch.argmax(q_values_masked))

            next_state, reward, done, _ = env.step(action)
            state_tensor = torch.tensor(next_state, dtype=torch.float32)
            total_r += reward

        rewards_log.append(total_r)
        # print(total_r, shortest)
        optimal_misses.append(int(abs(total_r - shortest) > 1e-3))
        if optimal_misses[-1] == 0 and found_episode == num_episodes:
            found_episode = episode

    # print(found_episode)

    return rewards_log, optimal_misses, found_episode