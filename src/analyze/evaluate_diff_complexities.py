import numpy as np
import matplotlib.pyplot as plt
from src.graph_env import create_graph, GraphEnv
from src.train import train_dqn_masked, train_actor_critic
from src.utils import perturb_graph
from src.evaluation import evaluate_on_perturbed


def robust_diff_complexities(
        node_range=range(5, 7),
        edge_range=None,
        num_episodes=350,
        num_perturbations=40,
):

    if edge_range is None:
        edge_range = node_range

    results = {
        "DQN": {},
        "FGSM": {},
        "A2C": {},
        "EAAN": {},
        "EACN": {},
        "FGSM_critic": {},
        "FGSM_actor": {}
    }

    for n in node_range:
        for e in range(n, n + 1):
            if e < (n - 1):
                continue

            print(f"\n=== Training on Graph with {n} nodes and {e} edges ===")

            G = create_graph(n, e)
            env = GraphEnv(G)

            model_dqn, _, _, _ = train_dqn_masked(env, num_episodes=num_episodes, use_fgsm=False)
            model_fgsm, _, _, _ = train_dqn_masked(env, num_episodes=num_episodes, use_fgsm=True)
            model_a2c, _, _, _ = train_actor_critic(env, num_episodes=num_episodes, attack=None)
            model_eaan, _, _, _ = train_actor_critic(env, num_episodes=num_episodes, attack='eaan')
            model_eacn, _, _, _ = train_actor_critic(env, num_episodes=num_episodes, attack='eacn')
            # model_fgsm_critic, _, _, _ = train_actor_critic(env, num_episodes=num_episodes, attack='fgsm_critic')
            # model_fgsm_actor, _, _, _ = train_actor_critic(env, num_episodes=num_episodes, attack='fgsm_actor')

            dqn_rewards = []
            fgsm_rewards = []
            a2c_rewards = []
            eaan_rewards = []
            eacn_rewards = []
            fgsm_critic_rewards = []
            fgsm_actor_rewards = []

            for _ in range(num_perturbations):
                G_perturbed = perturb_graph(G)
                r_dqn, _, dqn_ep = evaluate_on_perturbed(model_dqn, G_perturbed, GraphEnv, num_episodes=50)
                r_fgsm, _, dqn_fgsm_ep = evaluate_on_perturbed(model_fgsm, G_perturbed, GraphEnv, num_episodes=50)
                r_a2c, _, a2c_ep = evaluate_on_perturbed(model_a2c, G_perturbed, GraphEnv, num_episodes=50)
                r_eaan, _, a2c_eaan_ep = evaluate_on_perturbed(model_eaan, G_perturbed, GraphEnv, num_episodes=50)
                r_eacn, _, a2c_eacn_ep = evaluate_on_perturbed(model_eacn, G_perturbed, GraphEnv, num_episodes=50)
                # r_fgsm_critic, _, a2c_critic_ep = evaluate_on_perturbed(model_fgsm_critic, G_perturbed, GraphEnv, num_episodes=50)
                # r_fgsm_actor, _, a2c_actor_ep = evaluate_on_perturbed(model_fgsm_actor, G_perturbed, GraphEnv, num_episodes=50)

                dqn_rewards.append(np.mean(dqn_ep))
                fgsm_rewards.append(np.mean(dqn_fgsm_ep))
                a2c_rewards.append(np.mean(a2c_ep))
                eaan_rewards.append(np.mean(a2c_eaan_ep))
                eacn_rewards.append(np.mean(a2c_eacn_ep))
                # fgsm_critic_rewards.append(np.mean(a2c_critic_ep))
                # fgsm_actor_rewards.append(np.mean(a2c_actor_ep))

            # 4) Compute and store the average reward for each model
            results["DQN"][(n, e)] = np.mean(dqn_rewards)
            results["FGSM"][(n, e)] = np.mean(fgsm_rewards)
            results["A2C"][(n, e)] = np.mean(a2c_rewards)
            results["EAAN"][(n, e)] = np.mean(eaan_rewards)
            results["EACN"][(n, e)] = np.mean(eacn_rewards)
            # results["FGSM_critic"][(n, e)] = np.mean(fgsm_critic_rewards)
            # results["FGSM_actor"][(n, e)] = np.mean(fgsm_actor_rewards)

    return results

if __name__ == "__main__":
    node_list = range(5, 50)
    results_dict = run_perturbation_experiments(
        node_range=node_list,
        edge_range=None,
        num_episodes=350,
        num_perturbations=40
    )