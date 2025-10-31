import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import time
import os
import math
import random

def simulate_exploration(grid_size=15):
    """
    Simulates a robot creating a 2D grid map by autonomously exploring and
    avoiding obstacles. This version shows the robot as a blue circle with an
    arrow indicating its heading direction, moving in continuous paths.
    
    Press ESCAPE to stop the simulation and save the current map.

    NOTE: Live animation requires a compatible GUI backend (like Tkinter, Qt). If it fails,
    this script will run without animation and produce a final image file 'robot_map.png'.

    Args:
        grid_size (int): The dimensions of the square grid map (grid_size x grid_size).
    """
    
    # Global flag to track if simulation should stop
    stop_simulation = [False]  # Using list to allow modification in nested function

    # --- 1. Map Initialization and Legend Definition ---
    # 0: Unexplored/Empty Space (Light Gray)
    # 1: Obstacle (Black/Dark Gray)
    # 2: Explored Path (Green)
    map_data = np.zeros((grid_size, grid_size), dtype=int)
    colors = ['#e0e0e0', '#333333', '#4CAF50']
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

    # --- 3. Robot and Exploration Algorithm Setup ---
    # Robot state: position (continuous coordinates) and heading angle
    robot_x, robot_y = 1.5, 1.5  # Start at center of cell (1,1)
    robot_heading = 0  # Heading angle in radians (0 = right, π/2 = up, π = left, 3π/2 = down)
    robot_radius = 0.3
    
    # Using a stack for a Depth-First Search (DFS) exploration
    start_pos = (1, 1)
    stack = [start_pos]
    
    # --- 4. Matplotlib Visualization Setup ---
    fig, ax = plt.subplots(figsize=(10, 8))
    is_interactive = False
    im = None
    robot_circle = None
    robot_arrow = None

    try:
        plt.ion()
        im = ax.imshow(map_data, cmap=cmap, interpolation='none', vmin=0, vmax=2, extent=[0, grid_size, grid_size, 0])
        
        # Create robot visualization elements
        robot_circle = plt.Circle((robot_x, robot_y), robot_radius, color='blue', alpha=0.8, zorder=10)
        ax.add_patch(robot_circle)
        
        # Create arrow for heading direction
        arrow_length = 0.4
        arrow_end_x = robot_x + arrow_length * math.cos(robot_heading)
        arrow_end_y = robot_y - arrow_length * math.sin(robot_heading)  # Negative because y-axis is flipped
        robot_arrow = FancyArrowPatch((robot_x, robot_y), (arrow_end_x, arrow_end_y),
                                    arrowstyle='->', mutation_scale=15, color='white', linewidth=2, zorder=11)
        ax.add_patch(robot_arrow)
        
        ax.set_title("2D Robot Grid Map Exploration (Live) - Press ESCAPE to stop")
        ax.set_xlim(0, grid_size)
        ax.set_ylim(grid_size, 0)
        ax.set_aspect('equal')
        
        # Set tick marks at interval of 1
        ax.set_xticks(np.arange(0, grid_size + 1, 1))
        ax.set_yticks(np.arange(0, grid_size + 1, 1))
        ax.grid(True, alpha=0.3)
        
        # Add keyboard event handler
        def on_key_press(event):
            if event.key == 'escape':
                stop_simulation[0] = True
                print("\nESCAPE pressed! Stopping simulation and saving map...")
        
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
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
    target_r, target_c = current_r, current_c
    
    # Mark initial position as explored
    map_data[current_r, current_c] = 2

    if is_interactive:
        im.set_data(map_data)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.5) # Shorter initial pause

    print(f"Starting exploration from: ({current_r}, {current_c})")
    print("Press ESCAPE at any time to stop simulation and save the current map.")

    def save_current_map(current_map, robot_pos_x, robot_pos_y, robot_head, filename_prefix="interrupted"):
        """Save the current state of the map"""
        try:
            # Create a new figure for saving
            save_fig, save_ax = plt.subplots(figsize=(10, 8))
            
            # Display the current map
            save_ax.imshow(current_map, cmap=cmap, interpolation='none', vmin=0, vmax=2, extent=[0, grid_size, grid_size, 0])
            
            # Add robot position
            robot_save_circle = plt.Circle((robot_pos_x, robot_pos_y), robot_radius, color='blue', alpha=0.8, zorder=10)
            save_ax.add_patch(robot_save_circle)
            
            # Add robot heading arrow
            arrow_length = 0.4
            arrow_end_x = robot_pos_x + arrow_length * math.cos(robot_head)
            arrow_end_y = robot_pos_y - arrow_length * math.sin(robot_head)
            robot_save_arrow = FancyArrowPatch((robot_pos_x, robot_pos_y), (arrow_end_x, arrow_end_y),
                                             arrowstyle='->', mutation_scale=15, color='white', linewidth=2, zorder=11)
            save_ax.add_patch(robot_save_arrow)
            
            save_ax.set_title(f"Robot Grid Map - {filename_prefix.title()} State")
            save_ax.set_xlabel("X-Coordinate (Column Index)")
            save_ax.set_ylabel("Y-Coordinate (Row Index)")
            save_ax.set_xlim(0, grid_size)
            save_ax.set_ylim(grid_size, 0)
            save_ax.set_aspect('equal')
            
            # Add grid lines and ticks
            save_ax.set_xticks(np.arange(0, grid_size + 1, 1))
            save_ax.set_yticks(np.arange(0, grid_size + 1, 1))
            save_ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
            
            # Add legend
            legend_patches = [
                mpatches.Patch(color=colors[0], label='Unexplored (0)'),
                mpatches.Patch(color=colors[1], label='Obstacle (1)'),
                mpatches.Patch(color=colors[2], label='Explored Path (2)'),
                mpatches.Patch(color='blue', label='Robot (Blue Circle)')
            ]
            save_ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            
            # Save the file
            output_filename = f'robot_map_{filename_prefix}.png'
            save_fig.savefig(output_filename, bbox_inches='tight')
            print(f"\nMap saved successfully to '{os.path.abspath(output_filename)}'")
            
            # Close the save figure to free memory
            plt.close(save_fig)
            
        except Exception as e:
            print(f"\nERROR: Could not save the map. {e}")

    def update_robot_position(x, y, heading):
        """Update robot visual elements"""
        if is_interactive and robot_circle and robot_arrow:
            robot_circle.center = (x, y)
            
            # Update arrow direction
            arrow_length = 0.4
            arrow_end_x = x + arrow_length * math.cos(heading)
            arrow_end_y = y - arrow_length * math.sin(heading)  # Negative because y-axis is flipped
            robot_arrow.set_positions((x, y), (arrow_end_x, arrow_end_y))

    def calculate_heading(from_pos, to_pos):
        """Calculate heading angle from one position to another"""
        dx = to_pos[1] - from_pos[1]  # Column difference (x direction)
        dy = from_pos[0] - to_pos[0]  # Row difference (y direction, flipped)
        return math.atan2(dy, dx)

    def smooth_turn_and_move(start_x, start_y, target_x, target_y, start_heading, target_heading, steps=5):
        """Smoothly turn and move robot from start to target position - much faster movement"""
        # First, smoothly turn to face the target direction (much faster turning)
        turn_steps = 3
        for i in range(turn_steps + 1):
            if stop_simulation[0]:  # Check for escape during turning
                break
                
            t = i / turn_steps
            # Smooth interpolation between headings (handling angle wrapping)
            angle_diff = target_heading - start_heading
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            current_heading = start_heading + t * angle_diff
            update_robot_position(start_x, start_y, current_heading)
            
            if is_interactive:
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.005)  # Much faster turning
        
        # Then smoothly move to target position (much faster movement)
        for i in range(steps + 1):
            if stop_simulation[0]:  # Check for escape during movement
                break
                
            t = i / steps
            current_x = start_x + t * (target_x - start_x)
            current_y = start_y + t * (target_y - start_y)
            
            update_robot_position(current_x, current_y, target_heading)
            
            if is_interactive:
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.005)  # Much faster movement
        
        return target_x, target_y, target_heading

    while len(stack) > 0 and not stop_simulation[0]:
        # Check for escape key press during movement
        if stop_simulation[0]:
            break
            
        current_r, current_c = stack[-1] # Peek at current position from stack top

        # Define potential neighbors in a random order for randomness in movement
        neighbors = [
            (current_r + 1, current_c), # Down
            (current_r, current_c + 1), # Right
            (current_r - 1, current_c), # Up
            (current_r, current_c - 1)  # Left
        ]
        
        # Shuffle neighbors to add randomness in direction selection
        random.shuffle(neighbors)

        found_next_move = False
        for next_r, next_c in neighbors:
            # Check if neighbor is valid: within bounds and is an unexplored cell (value 0)
            if (0 <= next_r < grid_size and 
                0 <= next_c < grid_size and 
                map_data[next_r, next_c] == 0):

                # --- Calculate new heading and move forward ---
                target_r, target_c = next_r, next_c
                
                # Calculate heading angle for the movement
                new_heading = calculate_heading((current_r, current_c), (target_r, target_c))
                
                # Calculate target position (center of target cell)
                target_x, target_y = target_c + 0.5, target_r + 0.5
                
                # Smooth movement to target with turning
                robot_x, robot_y, robot_heading = smooth_turn_and_move(robot_x, robot_y, target_x, target_y, robot_heading, new_heading)
                
                # Mark the cell as explored as robot passes through
                map_data[target_r, target_c] = 2
                
                # Update current position
                current_r, current_c = target_r, target_c
                stack.append((current_r, current_c)) # Push new position to stack
                found_next_move = True
                
                # Update visualization
                if is_interactive:
                    im.set_data(map_data)
                    ax.set_title(f"Exploring... Position: ({current_r}, {current_c}) - Press ESCAPE to stop")
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.02)  # Much faster overall animation
                
                # Check for escape key press after movement
                if stop_simulation[0]:
                    break
                
                break # Move to the first valid neighbor

        # If no valid neighbor was found, the robot is at a dead end and must backtrack
        if not found_next_move:
            # --- Backtrack ---
            dead_end_r, dead_end_c = stack.pop() # Pop the dead-end position

            if len(stack) > 0: # If there's a path to backtrack to
                # The new current position is the one now at the top of the stack
                target_r, target_c = stack[-1]
                
                # Calculate heading for backtracking
                new_heading = calculate_heading((current_r, current_c), (target_r, target_c))
                
                # Calculate target position for backtracking
                target_x, target_y = target_c + 0.5, target_r + 0.5
                
                # Smooth movement back with turning
                robot_x, robot_y, robot_heading = smooth_turn_and_move(robot_x, robot_y, target_x, target_y, robot_heading, new_heading)
                
                # Update current position
                current_r, current_c = target_r, target_c

                # Update visualization during backtracking
                if is_interactive:
                    ax.set_title(f"Backtracking... Position: ({current_r}, {current_c}) - Press ESCAPE to stop")
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.02)  # Much faster backtracking animation
                
                # Check for escape key press during backtracking
                if stop_simulation[0]:
                    break

    # Check if simulation was stopped by user or completed naturally
    if stop_simulation[0]:
        print("Simulation stopped by user!")
        save_current_map(map_data, robot_x, robot_y, robot_heading, "interrupted")
    else:
        print("Exploration finished!")

    # --- 6. Final Plot Generation and Saving ---
    if not stop_simulation[0]:  # Only create final plot if not interrupted
        if is_interactive:
            plt.ioff()

        ax.clear()
        im_final = ax.imshow(map_data, cmap=cmap, interpolation='none', vmin=0, vmax=2, extent=[0, grid_size, grid_size, 0])
        
        # Add final robot position
        final_robot_circle = plt.Circle((robot_x, robot_y), robot_radius, color='blue', alpha=0.8, zorder=10)
        ax.add_patch(final_robot_circle)
        
        # Add final robot heading arrow
        arrow_length = 0.4
        arrow_end_x = robot_x + arrow_length * math.cos(robot_heading)
        arrow_end_y = robot_y - arrow_length * math.sin(robot_heading)
        final_robot_arrow = FancyArrowPatch((robot_x, robot_y), (arrow_end_x, arrow_end_y),
                                          arrowstyle='->', mutation_scale=15, color='white', linewidth=2, zorder=11)
        ax.add_patch(final_robot_arrow)
        
        ax.set_title("2D Robot Grid Map Exploration (Final State)")
        ax.set_xlabel("X-Coordinate (Column Index)")
        ax.set_ylabel("Y-Coordinate (Row Index)")
        ax.set_xlim(0, grid_size)
        ax.set_ylim(grid_size, 0)
        ax.set_aspect('equal')
        
        # Add grid lines
        ax.set_xticks(np.arange(0, grid_size + 1, 1))
        ax.set_yticks(np.arange(0, grid_size + 1, 1))
        ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
    
        legend_patches = [
            mpatches.Patch(color=colors[0], label='Unexplored (0)'),
            mpatches.Patch(color=colors[1], label='Obstacle (1)'),
            mpatches.Patch(color=colors[2], label='Explored Path (2)'),
            mpatches.Patch(color='blue', label='Robot (Blue Circle)')
        ]
        ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        output_filename = 'robot_map_enhanced_final.png'
        try:
            plt.savefig(output_filename, bbox_inches='tight')
            print(f"\nSUCCESS: Final map has been saved to '{os.path.abspath(output_filename)}'")
        except Exception as e:
            print(f"\nERROR: Could not save the file. {e}")

        plt.show()
    else:
        # Close the interactive plot if it was interrupted
        if is_interactive:
            plt.ioff()
            plt.close(fig)

if __name__ == "__main__":
    simulate_exploration(grid_size=15)