# Robot Mapper - 2D Grid Exploration Simulation

A Python-based robot simulation that demonstrates autonomous exploration and mapping of a 2D grid environment with obstacle avoidance using a depth-first search algorithm.

## Features

🤖 **Visual Robot Representation**
- Blue circle robot with directional arrow showing heading
- Smooth movement animations with realistic turning
- Real-time position tracking

🗺️ **Interactive Mapping**
- Live visualization of exploration progress
- Grid cells filled as robot explores them
- Clear distinction between unexplored areas, obstacles, and explored paths

⚡ **Fast & Responsive**
- Optimized animation speed
- Random direction selection for varied exploration patterns
- Keyboard controls for user interaction

🛑 **User Controls**
- Press **ESCAPE** at any time to stop simulation and save current map
- Automatic saving of final exploration results

## Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries
```bash
pip install numpy matplotlib
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

### Optional (for better performance)
```bash
pip install tk  # For better GUI backend support
```

## Installation

1. **Clone or download** the repository to your local machine
2. **Navigate** to the mapping directory:
   ```bash
   cd "path/to/robotics_code/mapping"
   ```
3. **Install dependencies**:
   ```bash
   pip install numpy matplotlib
   ```

## How to Run

### Method 1: Command Line
```bash
python robot_mapper_enhanced.py
```

### Method 2: Python Launcher (Windows)
```bash
py robot_mapper_enhanced.py
```

### Method 3: IDE
Open `robot_mapper_enhanced.py` in your favorite Python IDE (VS Code, PyCharm, etc.) and run it.

## Controls

| Key | Action |
|-----|--------|
| **ESC** | Stop simulation immediately and save current map |
| **Close Window** | End simulation normally |

## Understanding the Visualization

### Color Legend
- 🔲 **Light Gray**: Unexplored areas
- ⬛ **Dark Gray/Black**: Obstacles (walls)
- 🟢 **Green**: Explored path (where robot has been)
- 🔵 **Blue Circle**: Robot current position
- ➡️ **White Arrow**: Robot heading direction

### Grid System
- Coordinate system with tick marks at every integer (0, 1, 2, ...)
- X-axis represents columns, Y-axis represents rows
- Robot moves only in cardinal directions (North, South, East, West)
- 90-degree turns only

## Output Files

The simulation creates different output files depending on how it ends:

### Normal Completion
- **Filename**: `robot_map_enhanced_final.png`
- **When**: Exploration completes naturally (all reachable areas explored)

### User Interrupted (ESC pressed)
- **Filename**: `robot_map_interrupted.png`
- **When**: User presses ESC key during simulation

Both files include:
- Complete map of explored vs unexplored areas
- Final robot position and orientation
- Grid coordinates and legend
- High-resolution PNG format suitable for reports/presentations

## Algorithm Details

### Exploration Strategy
- **Depth-First Search (DFS)** with backtracking
- **Random direction selection** for varied exploration patterns
- **Obstacle avoidance** - robot automatically navigates around walls
- **Complete coverage** - explores all reachable areas

### Movement Mechanics
- **Smooth interpolated movement** between grid cells
- **Realistic turning animations** before movement
- **Continuous path visualization** as robot explores
- **Collision detection** with walls and obstacles

## Customization

### Grid Size
Modify the grid size by changing the parameter in the main function:
```python
if __name__ == "__main__":
    simulate_exploration(grid_size=20)  # Change from default 15
```

### Speed Adjustment
Modify animation speeds in the `smooth_turn_and_move` function:
- Increase `steps` for smoother but slower movement
- Decrease `plt.pause()` values for faster animation

### Obstacle Layout
Modify the environment setup section to create custom obstacle courses:
```python
# Add custom obstacles
map_data[row_start:row_end, col_start:col_end] = 1
```

## Troubleshooting

### "Interactive plotting failed" Warning
**Problem**: Animation doesn't show, only final image is saved
**Solutions**:
1. Install tkinter: `pip install tk`
2. Use Jupyter Notebook environment
3. Try different Python IDE with GUI support

### "Python was not found" Error
**Problem**: Command line doesn't recognize Python
**Solutions**:
1. Use `py` instead of `python`
2. Add Python to system PATH
3. Use full path to Python executable

### Slow Performance
**Problem**: Animation runs too slowly
**Solutions**:
1. Reduce grid size (e.g., `grid_size=10`)
2. Close other applications
3. Use faster computer/more RAM

### Module Import Errors
**Problem**: "No module named 'numpy'" or similar
**Solution**:
```bash
pip install numpy matplotlib
# or
pip install -r requirements.txt
```

## Technical Specifications

- **Language**: Python 3.7+
- **GUI Framework**: Matplotlib
- **Algorithms**: Depth-First Search (DFS)
- **Coordinate System**: 2D Cartesian grid
- **Animation**: Real-time matplotlib updates
- **File I/O**: PNG image output

## Examples

### Typical Output
The robot will:
1. Start at position (1,1) facing right
2. Randomly choose available directions
3. Move smoothly between grid cells
4. Turn to face new directions before moving
5. Mark explored areas in green
6. Backtrack when reaching dead ends
7. Continue until all reachable areas are explored

### Sample Exploration Pattern
```
Start → Move Right → Move Down → Hit Wall → 
Turn Left → Move Left → Turn Up → Move Up → 
Continue exploring...
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Test with a smaller grid size first
4. Check Python version compatibility

## License

This project is for educational purposes. Feel free to modify and adapt for your learning needs.

---

**Happy Exploring! 🤖🗺️**