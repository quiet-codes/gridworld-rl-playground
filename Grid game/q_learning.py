import random
from baseline_grid_exp import GridWorld

random.seed(42)
env = GridWorld()
alpha, gamma = 0.1, 0.95
epsilon, epsilon_min, epsilon_decay = 1.0, 0.05, 0.995
episodes = 5000
num_actions = env.action_space.n

# 25 grid positions x 4 actions
Q = {
    (r, c): [0.0] * num_actions
    for r in range(5)
    for c in range(5)
}

def choose_action(state, epsilon):
    # Explore: try a random action
    if random.random() < epsilon:
        return random.choice(env.actions)

    # Exploit: choose the action with the highest Q-value
    values = Q[state]
    best_value = max(values)

    # Random tie-breaking avoids always favouring action 0
    best_actions = [
        a for a, value in enumerate(values)
        if value == best_value
    ]
    return random.choice(best_actions)


# Train the agent
training_history=[]
# Train the agent
for episode in range(episodes):
    episode_reward =0
    observation, _ = env.reset()
    state = tuple(observation)

    while True:
        action = choose_action(state, epsilon)

        observation, reward, terminated, truncated, _ = env.step(action)
        next_state = tuple(observation)
        episode_reward += reward

        # No future value is added after the episode ends
        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * max(Q[next_state])

        # Q-learning update
        old_value = Q[state][action]
        Q[state][action] = old_value + alpha * (
            target - old_value
        )

        state = next_state

        if terminated or truncated:
            break

    # Gradually reduce exploration
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )
    training_history.append({
    "episode": episode + 1,
    "reward": episode_reward,
    "success": int(terminated),
    "steps": env.steps,
    "epsilon": epsilon,
})


# Inspect the learned policy
arrows = {
    0: "^",  # Up
    1: "v",  # Down
    2: "<",  # Left
    3: ">"   # Right
}

print("\nLearned Policy:")
print("S = Start | G = Goal | # = Wall")
print()

for r in range(5):
    row_display = []

    for c in range(5):
        state = (r, c)

        if state == env.start:
            symbol = "S"
        elif state == env.goal:
            symbol = "G"
        elif state in env.walls:
            symbol = "#"
        else:
            values = Q[state]
            best_action = max(
                range(num_actions),
                key=lambda a: values[a]
            )
            symbol = arrows[best_action]

        row_display.append(symbol)

    print("  ".join(row_display))


# Inspect Q-values at the starting position
print("\nExample Q-values:")
state = env.start
print("Start state:", state)

for action, value in enumerate(Q[state]):
    print(arrows[action], round(value, 3))


# Evaluate without random exploration
eval_episodes = 200
successes = 0
total_reward = 0
total_steps = 0

for _ in range(eval_episodes):
    observation, _ = env.reset()
    state = tuple(observation)

    while True:
        # Greedy policy: choose the best learned action
        values = Q[state]
        best_value = max(values)
        best_actions = [
            a for a, value in enumerate(values)
            if value == best_value
        ]
        action = random.choice(best_actions)

        observation, reward, terminated, truncated, _ = env.step(action)
        state = tuple(observation)
        total_reward += reward
        total_steps += 1

        if terminated or truncated:
            if terminated:
                successes += 1
            break

print("Training episodes:", episodes)
print("Evaluation episodes:", eval_episodes)
print("Success rate:", round(100 * successes / eval_episodes, 1), "%")
print("Average reward:", round(total_reward / eval_episodes, 2))
print("Average steps:", round(total_steps / eval_episodes, 2))


# Inspect action values around the suspicious wall
states_to_check = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 2),
]

action_names = ["Up", "Down", "Left", "Right"]

for state in states_to_check:
    print(f"\nState: {state}")

    for action, value in enumerate(Q[state]):
        print(f"  {action_names[action]:5s}: {value:.4f}")

window = 100

for i in range(0, len(training_history), window):
    batch = training_history[i:i + window]

    success_rate = (
        sum(item["success"] for item in batch)
        / len(batch) * 100
    )

    average_reward = (
        sum(item["reward"] for item in batch)
        / len(batch)
    )

    print(
        f"Episodes {batch[0]['episode']}-"
        f"{batch[-1]['episode']}: "
        f"Success={success_rate:.1f}%, "
        f"Avg reward={average_reward:.2f}"
    )

def test_greedy_policy(env, Q, max_steps=50):
    observation, _ = env.reset()
    state = tuple(observation)
    total_reward = 0

    print("\nGreedy Policy Test")
    print("Start:", state)

    for step in range(max_steps):
        values = Q[state]
        best_value = max(values)

        # Break ties consistently for this diagnostic test
        action = next(
            a for a, value in enumerate(values)
            if abs(value - best_value) < 1e-12
        )

        observation, reward, terminated, truncated, _ = env.step(action)
        next_state = tuple(observation)
        total_reward += reward

        print(
            f"Step {step + 1}: {state} -> "
            f"{['Up', 'Down', 'Left', 'Right'][action]} -> "
            f"{next_state}, reward={reward:.1f}"
        )

        state = next_state

        if terminated:
            print("SUCCESS: Goal reached!")
            print("Total reward:", round(total_reward, 2))
            return True

        if truncated:
            break

    print("FAILED: Goal not reached within the step limit.")
    print("Total reward:", round(total_reward, 2))
    return False


test_greedy_policy(env, Q)
