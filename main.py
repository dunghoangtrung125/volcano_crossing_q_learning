import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from parameters import *

actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
action_list = list(actions.keys())

# Initialize Q-table
q_table = np.zeros((map_rows, map_columns, len(actions)))

def is_valid(state):
    """Check if a state is valid within the grid"""
    x, y = state
    return 0 <= x < map_rows and 0 <= y < map_columns

def get_next_state(state, action):
    """Get the next state after taking an action."""
    x, y = state
    dx, dy = actions[action]
    next_state = (x + dx, y + dy)
    if is_valid(next_state):
        return next_state
    return state

def get_reward(state):
    """Get the reward for entering a state."""
    if state == goal_position:
        return goal_reward
    elif state in volcano_position:
        return volcano_penalty
    else:
        return step_penalty

# Training the agent
for episode in range(T):
    state = start_position
    num_steps = 0
    while state != goal_position:
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
        
        # Decay epsilon
        temp = epsilon * decay
        epsilon = max(min_epsilon, temp)
        # Move to the next state
        state = next_state

# Display Q-table
print("Trained Q-table:")
print(q_table)

# Visualize the policy and animate the agent
fig, ax = plt.subplots(figsize=(map_rows, map_columns))
ax.set_xlim(0, map_columns)
ax.set_ylim(0, map_rows)

# set up images
agent_img = mpimg.imread('img/agent.png')
img_display = None
volcano_img = mpimg.imread('img/volcano.png')
goal_img = mpimg.imread('img/goal.png')
start_img = mpimg.imread('img/start.png')


def visualize_policy():
    policy_grid = np.empty((map_rows, map_columns), dtype=str)
    for x in range(map_rows):
        for y in range(map_columns):
            if (x, y) == goal_position:
                policy_grid[x, y] = 'G'
            elif (x, y) in volcano_position:
                policy_grid[x, y] = 'L'
            else:
                best_action = np.argmax(q_table[x, y])
                policy_grid[x, y] = action_list[best_action][0].upper()
    print("\nOptimal Policy:")
    print(policy_grid)

    # Draw grid
    for x in range(map_rows):
        for y in range(map_columns):
            if (x, y) == start_position:
                ax.imshow(start_img, extent=[y, y + 1, map_rows - x - 1, map_rows - x], aspect='auto')
            elif (x, y) == goal_position:
                ax.imshow(goal_img, extent=[y, y + 1, map_rows - x - 1, map_rows - x], aspect='auto')
            elif (x, y) in volcano_position:
                ax.imshow(volcano_img, extent=[y, y + 1, map_rows - x - 1, map_rows - x], aspect='auto')
            else:
                ax.add_patch(patches.Rectangle((y, map_rows - x - 1), 1, 1, edgecolor="black", facecolor="white", fill=True))

    # Draw arrows for the policy
    for x in range(map_rows):
        for y in range(map_columns):
            if (x, y) != goal_position and (x, y) not in volcano_position:
                best_action = np.argmax(q_table[x, y])
                dx, dy = actions[action_list[best_action]]
                ax.arrow(y + 0.5, map_rows - x - 0.5, 0.3 * dy, -0.3 * dx, head_width=0.1, color="black")

# Animation function
def animate_policy():
    agent_path = [start_position]
    state = start_position
    animation_steps = 0
    while state != goal_position:
        if animation_steps == 100:
            break
        animation_steps += 1
        action_idx = np.argmax(q_table[state[0], state[1]])
        action = action_list[action_idx]
        next_state = get_next_state(state, action)
        agent_path.append(next_state)
        state = next_state

    def update(frame):
        x, y = agent_path[frame]
        global img_display
        if img_display:
            img_display.remove()
        img_display = ax.imshow(agent_img, extent=(y, y + 1, map_rows - x - 1, map_rows - x), zorder=10)
        return img_display

    ani = FuncAnimation(fig, update, frames=len(agent_path), interval=500, repeat=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.title('Volcano Crossing Q-learning Demo')
    plt.show()

visualize_policy()
animate_policy()