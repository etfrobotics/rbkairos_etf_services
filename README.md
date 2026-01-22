# rbkairos_etf_services

ROS package containing the RDDL planning stack for the Robotnik **RB-KAIROS** platform.
---

## Requirements

- **ROS 1 + catkin workspace** (commonly Melodic/Noetic)
- Typical ROS deps: `rospy`, `std_msgs`, `geometry_msgs`, `sensor_msgs`, etc.  
  (Exact dependencies are declared in `package.xml`.)

> If you’re working with the Robotnik RB-KAIROS stack, you’ll usually also have the relevant RB-KAIROS packages and robot bringup in your workspace.

---

## Install & build (catkin)

1) Create/enter a catkin workspace:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
```

2) Clone the repo:

```bash
git clone https://github.com/etfrobotics/rbkairos_etf_services.git
```

3) Install dependencies & build:

```bash
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```
