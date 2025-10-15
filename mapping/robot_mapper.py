import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import time
import os

def simulate_exploration(grid_size=15):
    """
    Simulates a robot creating a 2D grid map by autonomously exploring and
    avoiding obstacles. This version is optimized to run in environments like
    VS Code by saving the final map to a file.

    NOTE: Live animation requires a compatible GUI backend (like Tkinter, Qt). If it fails,
    this script will run without animation and produce a final image file 'robot_map.png'.

    Args:
        grid_size (int): The dimensions of the square grid map (grid_size x grid_size).
    """

    # --- 1. Map Initialization and Legend Definition ---
    # 0: Unexplored/Empty Space (Light Gray)
    # 1: Obstacle (Black/Dark Gray)
    # 2: Explored Path (Green)
    # 3: Robot's Current Position (Orange/Red)
    map_data = np.zeros((grid_size, grid_size), dtype=int)
    colors = ['#e0e0e0', '#333333', '#4CAF50', '#FF5722']
    cmap = ListedColormap(colors)

    # --- 2. Static Environment Setup ---
    # More complex obstacle course for a better demonstration
    map_data[2:5, 5] = 1
    map_data[5, 2:8] = 1
    map_data[10, 4:12] = 1
    map_data[7, 9:13] = 1
    map_data[1, 10] = 1
    map_data[2, 10] = 1

    # Border walls
    map_data[:, 0] = 1
    map_data[:, grid_size - 1] = 1
    map_data[0, :] = 1
    map_data[grid_size - 1, :] = 1

    # --- 3. Exploration Algorithm Setup ---
    # Using a stack for a Depth-First Search (DFS) exploration
    start_pos = (1, 1)
    stack = [start_pos]
    
    # --- 4. Matplotlib Visualization Setup ---
    fig, ax = plt.subplots(figsize=(8, 8))
    is_interactive = False
    im = None

    try:
        plt.ion()
        im = ax.imshow(map_data, cmap=cmap, interpolation='none', vmin=0, vmax=3)
        ax.set_title("2D Robot Grid Map Exploration (Live)")
        is_interactive = True
    except Exception as e:
        print("-------------------------------------------------------------------------------------")
        print(f"WARNING: Interactive plotting failed ({type(e).__name__}).")
        print("The simulation will run without live animation.")
        print("To see the animation, ensure you have a GUI backend like Tkinter installed (`pip install tk`)")
        print("or run this script in a Jupyter Notebook environment within VS Code.")
        print("-------------------------------------------------------------------------------------")

    # --- 5. Autonomous Exploration Loop ---
    current_r, current_c = start_pos
    map_data[current_r, current_c] = 3 # Mark initial robot position

    if is_interactive:
        im.set_data(map_data)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(1) # Start with a longer pause

    print(f"Starting exploration from: ({current_r}, {current_c})")

    while len(stack) > 0:
        current_r, current_c = stack[-1] # Peek at current position from stack top

        # Define potential neighbors in a specific order (e.g., Down, Right, Up, Left)
        neighbors = [
            (current_r + 1, current_c), # Down
            (current_r, current_c + 1), # Right
            (current_r - 1, current_c), # Up
            (current_r, current_c - 1)  # Left
        ]

        found_next_move = False
        for next_r, next_c in neighbors:
            # Check if neighbor is valid: within bounds and is an unexplored cell (value 0)
            if (0 <= next_r < grid_size and 
                0 <= next_c < grid_size and 
                map_data[next_r, next_c] == 0):

                # --- Move Forward ---
                map_data[current_r, current_c] = 2 # Mark old position as explored
                current_r, current_c = next_r, next_c # Update coordinates
                map_data[current_r, current_c] = 3 # Mark new position as robot
                stack.append((current_r, current_c)) # Push new position to stack
                found_next_move = True
                break # Move to the first valid neighbor

        # If no valid neighbor was found, the robot is at a dead end and must backtrack
        if not found_next_move:
            # --- Backtrack ---
            dead_end_r, dead_end_c = stack.pop() # Pop the dead-end position
            map_data[dead_end_r, dead_end_c] = 2 # Mark it as explored

            if len(stack) > 0: # If there's a path to backtrack to
                # The new current position is the one now at the top of the stack
                current_r, current_c = stack[-1] 
                map_data[current_r, current_c] = 3 # Mark it as the robot's position

        # --- Update Visualization at Each Step (Forward or Backtrack) ---
        if is_interactive:
            im.set_data(map_data)
            ax.set_title(f"Exploring... Position: ({current_r}, {current_c})")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01) # Short pause to make animation visible

    print("Exploration finished.")

    # --- 6. Final Plot Generation and Saving ---
    if is_interactive:
        plt.ioff()

    ax.clear()
    ax.imshow(map_data, cmap=cmap, interpolation='none', vmin=0, vmax=3)
    ax.set_title("2D Robot Grid Map Exploration (Final State)")
    ax.set_xlabel("X-Coordinate (Column Index)")
    ax.set_ylabel("Y-Coordinate (Row Index)")
    ax.set_xticks(np.arange(-.5, grid_size, 1), minor=False)
    ax.set_yticks(np.arange(-.5, grid_size, 1), minor=False)
    ax.set_xticklabels(np.arange(0, grid_size + 1, 1), minor=False)
    ax.set_yticklabels(np.arange(0, grid_size + 1, 1), minor=False)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
    
    legend_patches = [
        mpatches.Patch(color=colors[0], label='Unexplored (0)'),
        mpatches.Patch(color=colors[1], label='Obstacle (1)'),
        mpatches.Patch(color=colors[2], label='Explored Path (2)'),
        mpatches.Patch(color=colors[3], label='Robot Position (3)')
    ]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    output_filename = 'robot_map.png'
    try:
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"\nSUCCESS: Final map has been saved to '{os.path.abspath(output_filename)}'")
    except Exception as e:
        print(f"\nERROR: Could not save the file. {e}")

    plt.show()

if __name__ == "__main__":
    simulate_exploration(grid_size=15)

