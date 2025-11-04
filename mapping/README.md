# Robotics Mapping Package

A comprehensive collection of Python-based robotics simulations demonstrating different mapping and localization algorithms. This package includes two main implementations:

1. **Robot Mapper Enhanced** - 2D grid exploration with autonomous navigation
2. **EKF SLAM** - Extended Kalman Filter Simultaneous Localization and Mapping

## Project Overview

This package contains two advanced robotics mapping implementations:

### 1. Robot Mapper Enhanced (`robot_mapper_enhanced.py`)
An autonomous grid-based exploration system that demonstrates:
- **Depth-First Search (DFS) exploration** with intelligent backtracking
- **Real-time visualization** with smooth robot movement animations
- **Interactive controls** for stopping and saving exploration progress
- **Complete obstacle avoidance** and path planning

### 2. EKF SLAM (`ekf_slam.py`)
A complete implementation of Extended Kalman Filter SLAM that features:
- **Simultaneous Localization and Mapping** using probabilistic methods
- **Landmark-based mapping** with uncertainty quantification
- **Real-time state estimation** with covariance visualization
- **Multi-sensor data fusion** with noise modeling

---

## Features Summary

### Robot Mapper Enhanced Features

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

### EKF SLAM Features

🧭 **Probabilistic Localization**
- Extended Kalman Filter for optimal state estimation
- Continuous pose tracking with uncertainty quantification
- Real-time covariance ellipse visualization

🗺️ **Landmark-Based Mapping**
- Automatic landmark initialization and tracking
- Dynamic state vector augmentation for new landmarks
- Cross-correlation handling between robot and landmark states

📊 **Advanced Visualization**
- Live trajectory comparison (true vs estimated paths)
- Real-time uncertainty ellipses for robot and landmarks
- Error tracking plots for position and landmark accuracy

🔬 **Realistic Sensor Modeling**
- Range and bearing measurements with configurable noise
- Sensor range limitations and detection thresholds
- Proper noise propagation through motion and measurement models

⚡ **Optimized Performance**
- Efficient matrix operations using NumPy and SciPy
- Batch measurement processing for multiple landmarks
- Memory-optimized state vector management

## Requirements

### Python Version
- Python 3.7 or higher

### Required Libraries
```bash
pip install numpy>=1.20.0 matplotlib>=3.5.0 scipy>=1.7.0
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

### Additional Dependencies for EKF SLAM
```bash
pip install scipy  # For advanced matrix operations (block_diag)
```

### Optional (for better performance)
```bash
pip install tk  # For better GUI backend support
```

## Installation and Setup

1. **Clone or download** the repository to your local machine
2. **Navigate** to the mapping directory:
   ```cmd
   cd "c:\path\to\robotics_code\mapping"
   ```
3. **Install dependencies**:
   ```cmd
   pip install numpy>=1.20.0 matplotlib>=3.5.0 scipy>=1.7.0
   ```
   Or use requirements file:
   ```cmd
   pip install -r requirements.txt
   ```

## Execution Instructions

### Running Robot Mapper Enhanced

#### Method 1: Command Line (Windows)
```cmd
python robot_mapper_enhanced.py
```

#### Method 2: Python Launcher (Windows)
```cmd
py robot_mapper_enhanced.py
```

#### Method 3: Direct Execution
```cmd
py -3 robot_mapper_enhanced.py
```

#### Method 4: IDE Execution
Open `robot_mapper_enhanced.py` in VS Code, PyCharm, or any Python IDE and run it.

**Expected Runtime:** 2-5 minutes for complete exploration (depending on grid size)

### Running EKF SLAM

#### Method 1: Command Line (Windows)
```cmd
python ekf_slam.py
```

#### Method 2: Python Launcher (Windows)
```cmd
py ekf_slam.py
```

#### Method 3: With Custom Parameters
```cmd
python ekf_slam.py
# Edit the constants in the file to modify simulation parameters
```

**Expected Runtime:** 50 seconds of simulation time (real-time visualization)

## Program Controls and Interaction

### Robot Mapper Enhanced Controls

| Key | Action |
|-----|--------|
| **ESC** | Stop simulation immediately and save current map |
| **Close Window** | End simulation normally |

### EKF SLAM Controls

| Action | Result |
|--------|--------|
| **Close Window** | End simulation and display final results |
| **Automatic** | Simulation runs for 50 seconds automatically |

## Understanding the Visualizations

### Robot Mapper Enhanced Visualization

#### Color Legend
- 🔲 **Light Gray (#e0e0e0)**: Unexplored areas
- ⬛ **Dark Gray/Black (#333333)**: Obstacles (walls)
- 🟢 **Green (#4CAF50)**: Explored path (where robot has been)
- 🔵 **Blue Circle**: Robot current position
- ➡️ **White Arrow**: Robot heading direction

#### Grid System
- Integer coordinate system with tick marks (0, 1, 2, ...)
- X-axis represents columns, Y-axis represents rows
- Robot moves only in cardinal directions (North, South, East, West)
- 90-degree turns only with smooth interpolated movement

### EKF SLAM Visualization

#### Main Simulation Plot
- **Blue X markers**: True landmark positions (ground truth)
- **Blue solid line**: True robot path (ground truth)
- **Red solid line**: EKF estimated robot path
- **Blue circle**: Current true robot position
- **Red circle**: Current EKF estimated robot position
- **Red + markers**: EKF estimated landmark positions
- **Red dashed ellipses**: 95% confidence ellipses for uncertainties

#### Error Tracking Plots
- **Robot Position Error**: Real-time tracking of localization accuracy
- **Landmark Position Error**: Average error across all observed landmarks

## Output Files and Results

### Robot Mapper Enhanced Output

The simulation creates different output files depending on how it ends:

#### Normal Completion
- **Filename**: `robot_map_enhanced_final.png`
- **When**: Exploration completes naturally (all reachable areas explored)
- **Content**: Complete exploration map with final robot position

#### User Interrupted (ESC pressed)
- **Filename**: `robot_map_interrupted.png`
- **When**: User presses ESC key during simulation
- **Content**: Current exploration state when stopped

Both files include:
- High-resolution PNG format suitable for reports/presentations
- Complete map legend and coordinate system
- Final robot position and orientation
- Grid coordinates for accurate measurement

### EKF SLAM Output

#### Real-time Display
- **Live visualization**: Continuous updates during simulation
- **Interactive matplotlib window**: Can be resized and manipulated
- **Final state display**: Remains open after simulation completion

#### Console Output
- **Landmark initialization messages**: When new landmarks are detected
- **Time progression**: Current simulation time
- **State vector information**: Debugging information about state growth

## Algorithm Details and Implementation

### Robot Mapper Enhanced Algorithm

#### Exploration Strategy
- **Depth-First Search (DFS)** with intelligent backtracking
- **Random direction selection** among valid moves for exploration variety
- **Complete coverage guarantee** - explores all reachable areas
- **Memory-efficient** using stack-based navigation

#### Movement Mechanics
- **Smooth interpolated movement** between grid cells (5 steps per move)
- **Realistic turning animations** before movement (3 steps per turn)
- **Continuous path visualization** as robot explores
- **Collision detection** with walls and boundary checking
- **Fast animation** optimized for responsiveness (0.005s pause intervals)

### EKF SLAM Algorithm

#### Core Components

**1. Motion Model**
```python
# Circular motion with velocity and angular velocity
x_new = x - (v/ω) * sin(θ) + (v/ω) * sin(θ + ω*dt)
y_new = y + (v/ω) * cos(θ) - (v/ω) * cos(θ + ω*dt)
θ_new = θ + ω*dt
```

**2. Measurement Model**
```python
# Range and bearing to landmarks
range = sqrt((lx - rx)² + (ly - ry)²)
bearing = arctan2(ly - ry, lx - rx) - rθ
```

**3. State Vector Structure**
```python
# Dynamic state vector: [rx, ry, rθ, l1x, l1y, l2x, l2y, ...]
# Robot state: first 3 elements
# Landmark states: pairs of (x,y) coordinates
```

**4. Jacobian Calculations**
- **Motion Jacobian (G)**: ∂f/∂x for prediction step
- **Measurement Jacobian (H)**: ∂h/∂x for correction step
- **Noise Jacobians (V)**: For proper uncertainty propagation

#### Advanced Features
- **State Augmentation**: Automatic addition of new landmarks to state vector
- **Cross-correlation Handling**: Proper covariance matrix expansion
- **Batch Processing**: Multiple landmark measurements processed simultaneously
- **Noise Modeling**: Realistic sensor and motion noise parameters

## Customization and Configuration

### Robot Mapper Enhanced Customization

#### Grid Size Modification
```python
# In the main function call at the bottom of robot_mapper_enhanced.py
if __name__ == "__main__":
    simulate_exploration(grid_size=20)  # Change from default 15
```

#### Speed Adjustment
```python
# In the smooth_turn_and_move function:
- steps=5        # Increase for smoother movement (slower)
- plt.pause(0.005)  # Decrease for faster animation
```

#### Custom Obstacle Layout
```python
# In the environment setup section:
map_data[row_start:row_end, col_start:col_end] = 1  # Add obstacles
map_data[7:10, 3:6] = 1  # Example: rectangular obstacle
```

### EKF SLAM Customization

#### Simulation Parameters
```python
# At the top of ekf_slam.py
DT = 0.1          # Time step (smaller = more accurate, slower)
SIM_TIME = 50.0   # Total simulation time
STATE_SIZE = 3    # Robot state dimensions
LANDMARK_SIZE = 2 # Landmark state dimensions
```

#### Noise Tuning
```python
# Motion noise (affects prediction uncertainty)
R_CONTROL = np.diag([0.02, np.deg2rad(1.0)])**2

# Sensor noise (affects measurement uncertainty)
Q_SENSOR = np.diag([0.1, np.deg2rad(0.5)])**2
```

#### Landmark Configuration
```python
# Modify LANDMARKS_TRUE array to change landmark positions
LANDMARKS_TRUE = np.array([
    [5.0, 10.0],   # Landmark 1 at (5, 10)
    [10.0, 5.0],   # Landmark 2 at (10, 5)
    # Add more landmarks as needed
])
```

#### Motion Pattern
```python
# In get_control_input function
def get_control_input(time):
    v = 1.0      # Linear velocity (m/s)
    omega = 0.5  # Angular velocity (rad/s)
    return np.array([v, omega])
```

## Troubleshooting Guide

### Common Issues and Solutions

#### "Interactive plotting failed" Warning
**Problem**: Robot Mapper animation doesn't show, only final image is saved
**Solutions**:
1. Install tkinter: `pip install tk`
2. Use Jupyter Notebook environment in VS Code
3. Try different Python IDE with GUI support
4. Check matplotlib backend: `matplotlib.use('TkAgg')`

#### "Python was not found" Error
**Problem**: Command line doesn't recognize Python
**Solutions**:
1. Use `py` instead of `python`
2. Add Python to system PATH environment variable
3. Use full path to Python executable: `C:\Python39\python.exe script.py`

#### Slow Performance Issues
**Problem**: Animation runs too slowly or freezes
**Solutions**:
1. Reduce grid size (e.g., `grid_size=10` for Robot Mapper)
2. Close other resource-intensive applications
3. Reduce animation complexity:
   - Decrease `steps` parameter in movement functions
   - Increase `plt.pause()` values
4. Use faster computer with more RAM

#### Module Import Errors
**Problem**: "No module named 'numpy'", 'matplotlib', or 'scipy'
**Solutions**:
```cmd
pip install numpy matplotlib scipy
# or
pip install -r requirements.txt
# or for specific versions
pip install numpy>=1.20.0 matplotlib>=3.5.0 scipy>=1.7.0
```

#### EKF SLAM Specific Issues

**Problem**: "Singular matrix" or "LinAlgError"
**Cause**: Numerical instability in covariance matrix
**Solutions**:
1. Increase noise parameters (R_CONTROL, Q_SENSOR)
2. Ensure landmarks are well-separated
3. Check for duplicate landmark observations

**Problem**: Ellipses not displaying correctly
**Cause**: Matplotlib backend or covariance matrix issues
**Solutions**:
1. Update matplotlib: `pip install --upgrade matplotlib`
2. Check eigenvalue calculation in `plot_covariance_ellipse`
3. Verify positive definite covariance matrices

**Problem**: Memory usage increases over time
**Cause**: Growing state vector with many landmarks
**Solutions**:
1. Limit sensor range (modify detection threshold)
2. Implement landmark pruning for distant landmarks
3. Reduce simulation time or landmark density

## Technical Specifications and Performance

### Robot Mapper Enhanced
- **Language**: Python 3.7+
- **Dependencies**: NumPy, Matplotlib
- **Algorithm Complexity**: O(n) where n = number of reachable cells
- **Memory Usage**: O(grid_size²) for map storage
- **Animation Frame Rate**: ~200 FPS (5ms per frame)
- **Coordinate System**: 2D integer grid with continuous robot position

### EKF SLAM
- **Language**: Python 3.7+
- **Dependencies**: NumPy, Matplotlib, SciPy
- **Algorithm Complexity**: O(n³) where n = state vector size
- **Memory Usage**: O(n²) for covariance matrix storage
- **Real-time Performance**: Suitable for up to ~50 landmarks
- **Coordinate System**: Continuous 2D Cartesian coordinates
- **Numerical Precision**: 64-bit floating point (double precision)

## Educational Applications and Learning Outcomes

### Robot Mapper Enhanced Learning Objectives
- **Graph Search Algorithms**: Depth-First Search implementation
- **Robotics Fundamentals**: Grid-based navigation and path planning
- **Visualization Techniques**: Real-time matplotlib animation
- **Software Engineering**: Modular code design and error handling

### EKF SLAM Learning Objectives
- **Probabilistic Robotics**: Uncertainty representation and propagation
- **Linear Algebra**: Matrix operations, Jacobians, and eigenvalue decomposition
- **Sensor Fusion**: Combining motion and measurement information
- **State Estimation**: Kalman filtering theory and implementation
- **Computer Vision**: Landmark detection and association

## Research and Extension Opportunities

### Robot Mapper Enhanced Extensions
1. **Multi-robot exploration** with communication constraints
2. **Dynamic obstacle avoidance** with moving obstacles
3. **Optimal path planning** using A* or RRT algorithms
4. **3D grid exploration** for volumetric mapping
5. **Machine learning integration** for intelligent exploration strategies

### EKF SLAM Extensions
1. **Loop closure detection** for large-scale mapping
2. **Data association** for robust landmark matching
3. **Multi-robot SLAM** with map merging
4. **Visual SLAM** with camera-based features
5. **Particle Filter SLAM** for non-Gaussian uncertainty

## Support and Resources

For issues, questions, or further development:

1. **Check the troubleshooting section** above for common solutions
2. **Verify all dependencies** are installed with correct versions
3. **Test with smaller parameters** first (smaller grid size, shorter simulation time)
4. **Check Python version compatibility** (3.7+ required)
5. **Review console output** for specific error messages and debugging information

### Useful Resources
- **NumPy Documentation**: https://numpy.org/doc/
- **Matplotlib Documentation**: https://matplotlib.org/stable/
- **SciPy Documentation**: https://docs.scipy.org/
- **Probabilistic Robotics Book**: Thrun, Burgard, Fox (EKF SLAM reference)
- **Python Path Planning**: https://github.com/AtsushiSakai/PythonRobotics

## Example Usage Scenarios

### Robot Mapper Enhanced
```python
# Basic usage
simulate_exploration(grid_size=15)

# Custom configuration
simulate_exploration(grid_size=25)  # Larger exploration area
```

### EKF SLAM
```python
# Run with default parameters
python ekf_slam.py

# The simulation will automatically:
# 1. Initialize robot at origin (0,0,0)
# 2. Move in circular pattern for 50 seconds
# 3. Detect landmarks within 20m range
# 4. Update state estimate using EKF
# 5. Display real-time visualization
```

## License and Citation

This project is developed for educational purposes in robotics and computer science. Feel free to modify and adapt for your learning and research needs.

**When using this code for academic work, please cite:**
```
Robotics Mapping Package - 2D Grid Exploration and EKF SLAM Implementation
Educational Software for Autonomous Navigation and Probabilistic Robotics
[Your Institution/Course Information]
```

---

**Happy Exploring and Mapping! 🤖🗺️📊**