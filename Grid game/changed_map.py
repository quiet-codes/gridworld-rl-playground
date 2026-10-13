from baseline_grid_exp import GridWorld
from q_learning import Q

env = GridWorld()
env.walls.add((2, 2))
print("Changed map wall added at (2, 2)")
print("Testing the existing Q-table without retraining.")

observation, _ = env.reset()
state = tuple(observation)
total_reward = 0
terminated = truncated = False
for step in range(env.max_steps):
    values = Q[state]
    best_value = max(values)
    action = next(a for a, value in enumerate(values)
                  if abs(value - best_value) < 1e-12)
    observation, reward, terminated, truncated, _ = env.step(action)
    next_state = tuple(observation)
    total_reward += reward
    print(f"Step {step + 1}: {state} -> action {action} -> {next_state}, reward={reward:.1f}")
    state = next_state
    if terminated or truncated:
        break
print("Success:", terminated)
print("Total reward:", round(total_reward, 2))
print("Steps:", env.steps)
env.close()
