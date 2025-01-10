# Map set up
map_rows = 8  # Number of rows
map_columns = 8  # Number of columns
start_position = (0, 0)
goal_position = (map_rows - 1, map_columns - 1)
volcano_position = [(0, 2), (2, 2), (4, 2), (4, 3), (3, 0), (3,1), (4, 5), (4, 6), (4, 7), (6, 5)]

goal_reward = 20
volcano_penalty = -50
step_penalty = -1

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1  # Exploration rate
min_epsilon = 0.01
decay = 0.9999
T = 100000  # Number of episodes