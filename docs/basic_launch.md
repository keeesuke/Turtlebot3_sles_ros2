
Here are the basic commands to run/record turtlebot3 robot in real world.
This will cover how to run 1) MPC-only, 2) NN-only, 3) Switching

Before running those commands, the local pc needs environmental setting which can be set '. sles_env.sh'
make sure you are at the right directory: 'cd /home/acrl/sles/Turtlebot3_sles_ros2' and run above shell script whenever openning the new terminal

### Terminal 1 (SSH) — Bringup

```bash
ssh sigrobotics@10.195.154.171   # password: sigrobotics
. ~/.bashrc
ros2 launch turtlebot3_bringup robot.launch.py
```

### Terminal 2 (local) — Cartographer SLAM

Place the robot at your desired **start position** first, then launch SLAM:

```bash
. 
ros2 launch turtlebot3_cartographer cartographer.launch.py \
    use_sim_time:=false resolution:=0.02 publish_period_sec:=0.01
```

Try either logic below:
after running the command below, set the goal position on rviz2 from GoalPose icon. Once you set the goal, the robot begins to move.
### Terminal (local) — MPC

```bash
ros2 launch turtlebot3_sles_control turtlebot3_planner_real_world.launch.py
```

### Terminal (local) — NN

```bash
ros2 launch turtlebot3_sles_control turtlebot3_planner_NN_real_world.launch.py     model_path:=~/robot_data/real_world_models/run01/best_model.pth
```

### Terminal (local) — Switch

```bash
ros2 launch turtlebot3_sles_control turtlebot3_planner_switch_MPC_NN_real_world.launch.py model_path:=~/robot_data/real_world_models/run01/best_model.pth
```


To record the whole data for each logic, try this:
```bash
ros2 run turtlebot3_sles_data experiment_recorder.py --logic mpc
```
Depending on which logic you are going to run, 'logic' augment accepts either mpc|nn|switch

