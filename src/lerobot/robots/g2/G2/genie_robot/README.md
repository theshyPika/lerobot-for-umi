# Genie Robot Package

This is a ROS2 package for displaying and controlling the Genie robot.

## Usage

### 1. Build Package

```bash
# In workspace root directory
colcon build --packages-select genie_robot
source install/setup.bash
```

### 2. Launch Robot Display

#### Basic Display
```bash
ros2 launch genie_robot display.launch.py
```

## File Structure

```
genie_robot/
├── launch/                        # Launch files
│   └── display.launch.py    # Basic display
├── urdf/                          # URDF model files
│   └── G2_t2_crs/                 # G2 T2 CRS configuration
├── meshes/                        # 3D mesh files
│   ├── G2/                        # G2 series meshes
│   └── arm/                       # robot arm meshes
├── rviz/                          # RViz configuration files
│   └── default.rviz               # Default RViz configuration
├── package.xml                    # Package dependencies
└── CMakeLists.txt                 # Build configuration
```

## Dependencies

- ROS2 (Humble or later)
- joint_state_publisher
- robot_state_publisher