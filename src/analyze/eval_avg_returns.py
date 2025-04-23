import matplotlib.pyplot as plt
import networkx as nx
from src.graph_env import GraphEnv, create_graph
from src.train import train_dqn_masked, train_actor_critic
from src.utils import moving_average

# function to plot average rewards for past [window size] episodes
def evaluate_plot_returns(n_nodes, n_edges):
    G = create_graph(n_nodes, n_edges)
    env = GraphEnv(G)

    optimal_cost = -nx.dijkstra_path_length(env.our_graph, env.start_node, env.goal_node, weight='weight') + 100

    print("optimal cost according to dijkstra is :", optimal_cost)

    print("we train DQN without an attack")
    model_dqn, rewards_dqn, misses_dqn, _ = train_dqn_masked(env, num_episodes=400, use_fgsm=False)
    print("we train DQN attacked with FGSM")
    model_dqn_fgsm, rewards_fgsm, misses_fgsm, _ = train_dqn_masked(env, num_episodes=400 ,use_fgsm=True)
    print("we train A2C without an attack")
    model_a2c, rewards_a2c, misses_a2c, _ = train_actor_critic(env, num_episodes=400,attack=None)
    print("we train A2C attacked with EAAN")
    model_eaan, rewards_eaan, misses_eaan, _ = train_actor_critic(env, num_episodes=400,attack='eaan')
    print("we train A2C attacked with EACN")
    model_eacn, rewards_eacn, misses_eacn, _ = train_actor_critic(env, num_episodes=400,attack='eacn')
    # print("we train A2C attacked with FGSM on Critic")
    # model_fgsm_critic, rewards_fgsm_critic, misses_fgsm_critic, _ = train_actor_critic(env, num_episodes=350,attack='fgsm_critic')
    # print("we train A2C attacked with FGSM on Actor")
    # model_fgsm_actor, rewards_fgsm_actor, misses_fgsm_actor, _ = train_actor_critic(env, num_episodes=350,attack='fgsm_actor')
    # print("we train DQN with domain randomization")
    # model_dqn_dr, rewards_dqn_dr, misses_dqn_dr, _ = train_dqn_masked(env, num_episodes=500, use_fgsm=False, domain_randomization=True)
    # print("we train A2C with domain randomization")
    # model_a2c_dr, rewards_a2c_dr, misses_a2c_dr, _ = train_actor_critic(env, num_episodes=500, attack=None, domain_randomization=True)

    # we do not look on rewards for environment_randomized because graph changes
    # training rewards
    plt.figure()
    plt.plot(moving_average(rewards_dqn, 20), label="DQN")
    plt.plot(moving_average(rewards_fgsm, 20), label="DQN_FGSM")
    plt.plot(moving_average(rewards_a2c, 20), label="A2C")
    plt.plot(moving_average(rewards_eaan, 20), label="A2C_EAAN")
    plt.plot(moving_average(rewards_eacn, 20), label="A2C_EACN")
    # plt.plot(moving_average(rewards_fgsm_critic, 20), label="A2C_FGSM_Critic")
    # plt.plot(moving_average(rewards_fgsm_actor, 20), label="A2C_FGSM_Actor")

    plt.axhline(optimal_cost, color='black', linestyle='--', label='optimal dijkstra')

    plt.title("mean_average_rewards, window_size = 20")
    plt.xlabel("episode")
    plt.ylabel("mean_average_reward")
    plt.legend()
    plt.savefig("rewards_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    num_nodes, num_edges = map(int, input().split())
    evaluate_plot_returns(num_nodes, num_edges)