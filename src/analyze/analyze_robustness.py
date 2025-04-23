from src.graph_env import create_graph, GraphEnv, create_graph
from src.train import train_dqn_masked, train_actor_critic
from src.utils import perturb_graph
from scipy import stats
from src.evaluation import evaluate_on_perturbed
import networkx as nx
import numpy as np

def evaluate_robustness(num_graphs=100, num_nodes=15, num_edges=15):
    results = {
        'DQN': [], 'FGSM': [], 'A2C': [], 'EAAN': [], 'EACN': [], 'FGSM_critic': [], 'FGSM_actor': [], 'DQN_DR': [], 'A2C_DR': [],
        'DQN_dist': [], 'FGSM_dist': [], 'A2C_dist': [], 'EAAN_dist': [], 'EACN_dist': [], 'FGSM_critic_dist': [], 'FGSM_actor_dist': [],
        'DQN_DR_dist': [], 'A2C_DR_dist': []
    }

    # define the first_episode when we found an optimal path on train and on test/evaluation
    first_episode = {'DQN': [], 'FGSM': [], 'A2C': [], 'EAAN': [], 'EACN': [], 'FGSM_critic': [], 'FGSM_actor': [], 'DQN_DR' : [], 'A2C_DR': []}
    first_episode_perturb = {'DQN': [], 'FGSM': [], 'A2C': [], 'EAAN': [], 'EACN': [], 'FGSM_critic': [], 'FGSM_actor': [], 'DQN_DR' : [], 'A2C_DR': []}
    optimal_on_perturbed = {'DQN': 0, 'FGSM': 0, 'A2C': 0, 'EAAN': 0, 'EACN': 0, 'FGSM_critic': 0, 'FGSM_actor': 0, 'DQN_DR' : 0, 'A2C_DR': 0}
    for i in range(num_graphs):
        print(f"graph {i+1}/{num_graphs}")
        G = create_graph(num_nodes, num_edges)
        env = GraphEnv(G)

        model_dqn, _, _, dqn_ep = train_dqn_masked(env, num_episodes=400, use_fgsm=False)
        model_fgsm, _, _, dqn_fgsm_ep = train_dqn_masked(env, num_episodes=400, use_fgsm=True)
        model_a2c, _, _, a2c_ep = train_actor_critic(env, num_episodes=400, attack=None)
        model_eaan, _, _, a2c_eaan_ep = train_actor_critic(env, num_episodes=400, attack='eaan')
        model_eacn, _, _, a2c_eacn_ep = train_actor_critic(env, num_episodes=400,attack='eacn')
        # model_fgsm_critic, _, _, a2c_critic_ep = train_actor_critic(env, num_episodes=1000,attack='fgsm_critic')
        # model_fgsm_actor, _, _, a2c_actor_ep = train_actor_critic(env, num_episodes=1000,attack='fgsm_actor')
        model_dqn_dr, _, _, dqn_dr_ep = train_dqn_masked(env, num_episodes=400, use_fgsm=False, domain_randomization=True)
        model_a2c_dr, _, _, a2c_dr_ep = train_actor_critic(env, num_episodes=400, attack=None, domain_randomization=True)

        first_episode['DQN'].append(dqn_ep)
        first_episode['FGSM'].append(dqn_fgsm_ep)
        first_episode['A2C'].append(a2c_ep)
        first_episode['EAAN'].append(a2c_eaan_ep)
        first_episode['EACN'].append(a2c_eacn_ep)
        # first_episode['FGSM_critic'].append(a2c_critic_ep)
        # first_episode['FGSM_actor'].append(a2c_actor_ep)
        first_episode['DQN_DR'].append(dqn_dr_ep)
        first_episode['A2C_DR'].append(a2c_dr_ep)

        G_perturbed = perturb_graph(G, num_changes=num_nodes//2)
        # G_perturbed = create_graph(num_nodes, num_edges)
        opt_reward = -nx.dijkstra_path_length(G_perturbed, 0, num_nodes-1, weight='weight') + 100

        # FGSM_critic, FGSM_actor
        for label, model in zip(['DQN', 'FGSM', 'A2C', 'EAAN', 'EACN', 'DQN_DR', 'A2C_DR'],
                    [model_dqn, model_fgsm, model_a2c, model_eaan, model_eacn, model_dqn_dr, model_a2c_dr]):
            # we define num_episodes as 100 to find the optimal path
            rewards, _, found_episode = evaluate_on_perturbed(model, G_perturbed, GraphEnv, num_episodes=100)
            first_episode_perturb[label].append(found_episode)
            # avg_reward = np.mean(rewards)
            avg_reward = max(rewards)
            print(f'Using {label} maximum reward was {avg_reward} and Dijkstra was {opt_reward}')
            results[label].append(avg_reward)
            results[f"{label}_dist"].append(abs(avg_reward - opt_reward))
            if avg_reward == opt_reward:
                optimal_on_perturbed[label] += 1


    print(f"On the training first episode when was founded an optimal path")
    means = {key: np.mean(value) if value else 0 for key, value in first_episode.items()}
    print(means)

    print(f"On the test first episode when was founded an optimal path")
    print(first_episode_perturb)
    means_perturb = {key: np.mean(value) if value else 0 for key, value in first_episode_perturb.items()}
    print(means_perturb)

    for key in optimal_on_perturbed:
        optimal_on_perturbed[key] /= num_graphs
    print(f'Optimal path found on perturbed in % cases')
    print(optimal_on_perturbed)

    ci_rewards = {'DQN':0, 'FGSM':0, 'A2C':0, 'EAAN':0, 'EACN':0, 'DQN_DR': 0, 'A2C_DR': 0}
    ci_avg_distance_to_dijkstra = {'DQN':0, 'FGSM':0, 'A2C':0, 'EAAN':0, 'EACN':0, 'DQN_DR': 0, 'A2C_DR': 0}
    for label in ['DQN', 'FGSM', 'A2C', 'EAAN', 'EACN', 'DQN_DR', 'A2C_DR']:
        mean_r = np.mean(results[label])
        mean_d = np.mean(results[f"{label}_dist"])
        ci_r = stats.t.interval(0.95, len(results[label])-1, loc=mean_r, scale=stats.sem(results[label]))
        ci_d = stats.t.interval(0.95, len(results[f"{label}_dist"])-1, loc=mean_d, scale=stats.sem(results[f"{label}_dist"]))

        print(f"\n{label} avg reward: {mean_r:.2f} CI95: {ci_r}")
        print(f"{label} avg distance to dijkstra: {mean_d:.2f} CI95: {ci_d}")
        ci_rewards[label] = (mean_r, ci_r)
        ci_avg_distance_to_dijkstra[label] = (mean_d, ci_d)
    return means, means_perturb, ci_rewards, ci_avg_distance_to_dijkstra, optimal_on_perturbed

if __name__ == "__main__":
    print(evaluate_robustness(num_graphs=8, num_nodes=9, num_edges=9))