"""Gymnasium 5x5 GridWorld and random-agent baseline."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorld(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, max_steps=50):
        super().__init__()
        self.start, self.goal = (0, 0), (4, 4)
        self.walls = {(0, 3), (1, 1), (1, 3), (2, 3), (3, 0), (3, 1)}
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([5, 5])
        self.actions = list(range(4))  # Keep the tabular learner interface.
        self.position, self.steps = self.start, 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position, self.steps = self.start, 0
        return np.asarray(self.position, dtype=np.int64), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        r, c = self.position
        dr, dc = moves[int(action)]
        candidate = (r + dr, c + dc)
        self.steps += 1
        if (0 <= candidate[0] < 5 and 0 <= candidate[1] < 5
                and candidate not in self.walls):
            self.position = candidate
        terminated = self.position == self.goal
        reward = 1.0 if terminated else -0.1
        truncated = self.steps >= self.max_steps and not terminated
        return np.asarray(self.position, dtype=np.int64), reward, terminated, truncated, {}

def run_random_episode(env):
    _, _ = env.reset()
    total = 0.0
    while True:
        _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        total += reward
        if terminated or truncated:
            return total, terminated

def evaluate_random_agent(episodes=100):
    env = GridWorld()
    results = [run_random_episode(env) for _ in range(episodes)]
    print("Random-agent episodes:", episodes)
    print("Success rate:", round(100 * sum(ok for _, ok in results) / episodes, 1), "%")
    print("Average return:", round(sum(score for score, _ in results) / episodes, 2))
    env.close()

if __name__ == "__main__":
    evaluate_random_agent()
