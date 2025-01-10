import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg

# Initialize environment parameters
map_rows = 5
map_columns = 5

grid_size = 5  # 5x5 grid
start = (0, 0)  # Start position
goal = (0, 4)  # Goal position
lava = [(0, 2), (2, 2), (4, 2), (4, 3), (3,0)]  # Lava positions

# Define actions and their effects
actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
action_list = list(actions.keys())

# Initialize Q-table
q_table = np.zeros((map_rows, map_columns, len(actions)))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1  # Exploration rate
min_epsilon = 0.01
decay = 0.9999
n_episodes = 100000  # Number of episodes

# Helper functions
def is_valid(state):
    """Check if a state is valid within the grid and not in lava."""
    x, y = state
    return 0 <= x < grid_size and 0 <= y < grid_size

def get_next_state(state, action):
    """Get the next state after taking an action."""
    x, y = state
    dx, dy = actions[action]
    next_state = (x + dx, y + dy)
    if is_valid(next_state):
        return next_state
    return state  # If invalid move, stay in the same state

def get_reward(state):
    """Get the reward for entering a state."""
    if state == goal:
        return 50
    elif state in lava:
        return -50
    else:
        return -1

# Training the agent
for episode in range(n_episodes):
    state = start
    num_steps = 0
    while state != goal:
        if num_steps == 1000:
            break
        num_steps += 1
        # Choose action (epsilon-greedy)
        if np.random.random() <= epsilon:
            action_idx = random.randint(0, len(action_list) - 1)
        else:
            action_idx = np.argmax(q_table[state[0], state[1]])
        action = action_list[action_idx]

        # Take action and observe next state and reward
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)

        # Update Q-value
        best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
        td_target = reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - q_table[state[0], state[1], action_idx]
        q_table[state[0], state[1], action_idx] += alpha * td_error
        
        # decay epsilon
        temp = epsilon * decay
        epsilon = max(min_epsilon, temp)
        # Move to the next state
        state = next_state

# Display Q-table
print("Trained Q-table:")
print(q_table)

# Visualize the policy and animate the agent
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)

agent_img = mpimg.imread('img/agent.png')  # Replace with the path to your image file
img_display = None
volcano_img = mpimg.imread('img/volcano.png')
goal_img = mpimg.imread('img/goal.png')
start_img = mpimg.imread('img/start.png')

# Visualize the policy
def visualize_policy():
    policy_grid = np.empty((grid_size, grid_size), dtype=str)
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) == goal:
                policy_grid[x, y] = 'G'
            elif (x, y) in lava:
                policy_grid[x, y] = 'L'
            else:
                best_action = np.argmax(q_table[x, y])
                policy_grid[x, y] = action_list[best_action][0].upper()
    print("\nOptimal Policy:")
    print(policy_grid)

    # Draw grid
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) == start:
                # ax.add_patch(patches.Rectangle((y, grid_size - x - 1), 1, 1, color="green", alpha=0.5))
                # ax.text(y + 0.5, grid_size - x - 0.5, "S", ha="center", va="center", fontsize=14)
                ax.imshow(start_img, extent=[y, y + 1, grid_size - x - 1, grid_size - x], aspect='auto')
            elif (x, y) == goal:
                # ax.add_patch(patches.Rectangle((y, grid_size - x - 1), 1, 1, color="blue", alpha=0.5))
                # ax.text(y + 0.5, grid_size - x - 0.5, "G", ha="center", va="center", fontsize=14)
                ax.imshow(goal_img, extent=[y, y + 1, grid_size - x - 1, grid_size - x], aspect='auto')
            elif (x, y) in lava:
                # ax.add_patch(patches.Rectangle((y, grid_size - x - 1), 1, 1, color="red", alpha=0.5))
                ax.imshow(volcano_img, extent=[y, y + 1, grid_size - x - 1, grid_size - x], aspect='auto')
            else:
                ax.add_patch(patches.Rectangle((y, grid_size - x - 1), 1, 1, edgecolor="black", facecolor="white", fill=True))

    # Draw arrows for the policy
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) != goal and (x, y) not in lava:
                best_action = np.argmax(q_table[x, y])
                dx, dy = actions[action_list[best_action]]
                ax.arrow(y + 0.5, grid_size - x - 0.5, 0.3 * dy, -0.3 * dx, head_width=0.1, color="black")

# Animation function
def animate_policy():
    agent_path = [start]
    state = start
    while state != goal:
        action_idx = np.argmax(q_table[state[0], state[1]])
        action = action_list[action_idx]
        next_state = get_next_state(state, action)
        agent_path.append(next_state)
        state = next_state

    # agent_dot, = ax.plot(start[1] + 0.5, grid_size - start[0] - 0.5, 'go', markersize=10)

    def update(frame):
        x, y = agent_path[frame]
        # agent_dot.set_data(y + 0.5, grid_size - x - 0.5)
        # return agent_dot,
        global img_display
        if img_display:
            img_display.remove()
        img_display = ax.imshow(agent_img, extent=(y, y + 1, grid_size - x - 1, grid_size - x), zorder=10)
        return img_display,

    ani = FuncAnimation(fig, update, frames=len(agent_path), interval=500, repeat=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.xticks(range(grid_size + 1))
    plt.yticks(range(grid_size + 1))
    plt.title('Volcano Crossing Demo')
    plt.show()

visualize_policy()
animate_policy()
