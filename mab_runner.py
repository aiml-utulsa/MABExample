import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from mab import Bandit, GreedyBandit, EpsilonGreedyBandit, UCB1Bandit, RandomBandit


def run_bandit(
    bandit: Bandit,
    n_rounds,
    n_trials,
    stationary=True,
    means=np.random.random(5) * 2,
    sdevs=np.ones(5),
    deltas=np.zeros(5),
):
    selected = np.zeros(
        (
            n_trials,
            n_rounds,
            bandit.n_arms,
        )
    )
    regret = np.zeros((n_trials, n_rounds))
    estimates = np.zeros((n_trials, n_rounds, bandit.n_arms))

    for t in range(n_trials):
        bandit.reset()
        for round in range(n_rounds):
            arm = bandit.select_arm()
            reward = reward_function(means, sdevs, round, deltas)[arm]
            bandit.update(arm, reward)
            selected[t, round, arm] = 1
            regret[t, round] = np.max(means + deltas) - means[arm] + deltas[arm]
            estimates[t, round] = bandit.q_values.copy()

    return selected, regret, estimates


def reward_function(means=np.zeros(5), sdevs=np.ones(5), t=0, deltas=np.zeros(5)):
    return np.random.normal(
        loc=means + deltas * t, scale=sdevs
    )  # Replace with your own reward function


def plot_results(selected, regret, estimates, bandit_name):
    # Plotting the results
    n_trials, n_rounds, n_arms = selected.shape
    avg_regret = np.mean(regret, axis=0)
    avg_selected = np.mean(selected, axis=0)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(avg_regret)
    plt.title(f"{bandit_name} Average Regret")
    plt.xlabel("Rounds")
    plt.ylabel("Regret")
    # set ymin to zero
    plt.ylim(0, np.max(avg_regret) * 1.1)
    plt.grid()

    plt.subplot(1, 3, 2)
    for arm in range(n_arms):
        plt.plot(avg_selected[:, arm], label=f"Arm {arm}")
    plt.title(f"{bandit_name} Arm Selection")
    plt.xlabel("Rounds")
    plt.ylabel("Selection Probability")
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    for arm in range(n_arms):
        plt.plot(np.mean(estimates[:, :, arm], axis=0), label=f"Arm {arm}")
    plt.title(f"{bandit_name} Estimated Values")
    plt.xlabel("Rounds")
    plt.ylabel("Estimated Value")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_reward_dist(means, sdevs):
    plt.figure(figsize=(8, 4))
    for i in range(len(means)):
        x = np.linspace(means[i] - 3 * sdevs[i], means[i] + 3 * sdevs[i], 100)
        y = np.exp(-0.5 * ((x - means[i]) / sdevs[i]) ** 2) / (
            sdevs[i] * np.sqrt(2 * np.pi)
        )
        plt.plot(x, y, label=f"Arm {i}")

    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    n_rounds = 500
    n_trials = 200
    n_arms = 5

    means = np.random.random(n_arms) * 2 - 1
    means[0] = -1
    sdevs = np.ones(n_arms)
    deltas = np.zeros(n_arms)

    deltas[0] = 0.01
    print(f"Means: {means}\nStandard Deviations: {sdevs}\nDeltas: {deltas}")
    plot_reward_dist(means, sdevs)

    bandits = {
        "Random": RandomBandit(n_arms),
        # "Greedy": GreedyBandit(n_arms),
        "Epsilon-Greedy": EpsilonGreedyBandit(n_arms),
        "UCB1": UCB1Bandit(n_arms),
    }

    input("continue?")
    for name, bandit in bandits.items():
        selected, regret, estimates = run_bandit(
            bandit, n_rounds, n_trials, means=means, sdevs=sdevs, deltas=deltas
        )
        plot_results(selected, regret, estimates, name)
