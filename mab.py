import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Bandit(ABC):
    def __init__(self, n_arms, true_rewards):
        self.n_arms = n_arms
        self.true_rewards = true_rewards

    @abstractmethod
    def select_arm(self):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass

    @abstractmethod
    def reset(self):
        pass


class RandomBandit(Bandit):
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.true_rewards = np.zeros(n_arms)
        self.q_values = np.zeros(n_arms) + 1

    def select_arm(self):
        return np.random.randint(self.n_arms)

    def update(self, arm, reward):
        pass  # No update needed for random bandit

    def reset(self):
        self.true_rewards = np.zeros(self.n_arms)


class GreedyBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.reward_sum = np.zeros(n_arms)
        self.arm_counts = np.ones(n_arms)
        self.q_values = np.zeros(n_arms) + 100
        self.total_counts = 1

    def select_arm(self):
        return np.argmax(self.q_values)

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.total_counts += 1
        self.reward_sum[arm] += reward
        self.q_values[arm] = self.reward_sum[arm] / self.arm_counts[arm]

    def reset(self):
        self.reward_sum = np.zeros(self.n_arms)
        self.arm_counts = np.ones(self.n_arms)
        self.q_values = np.zeros(self.n_arms)
        self.total_counts = 0


class EpsilonGreedyBandit(Bandit):
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.reward_sum = np.zeros(n_arms)
        self.arm_counts = np.ones(n_arms)
        self.q_values = np.zeros(n_arms) + 5
        self.total_counts = 1

    def select_arm(self):
        epsilon = 0.1
        if np.random.rand() < epsilon * 10 / (self.total_counts + 1):
            return np.random.randint(self.n_arms)
        else:
            # Exploit
            return np.argmax(self.q_values)
        return np.argmax(self.q_values)

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.total_counts += 1
        self.reward_sum[arm] += reward
        self.q_values[arm] = 0.5 * self.q_values[arm] + 0.5 * reward

    def reset(self):
        self.reward_sum = np.zeros(self.n_arms)
        self.arm_counts = np.ones(self.n_arms)
        self.q_values = np.zeros(self.n_arms)
        self.total_counts = 0


class UCB1Bandit(Bandit):
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.reward_sum = np.zeros(n_arms)
        self.arm_counts = np.ones(n_arms)
        self.q_values = np.zeros(n_arms) + 5
        self.total_counts = 0

    def select_arm(self):
        return np.argmax(self.q_values)

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        self.total_counts += 1
        self.reward_sum[arm] += reward
        self.q_values[arm] = self.reward_sum[arm] / self.arm_counts[arm]

    def reset(self):
        self.reward_sum = np.zeros(self.n_arms)
        self.arm_counts = np.ones(self.n_arms)
        self.q_values = np.zeros(self.n_arms)
        self.total_counts = 0
