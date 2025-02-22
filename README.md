
# grid-mapping-in-ROS
Creating Occupancy Grid Maps using Static State Bayes filter and Bresenham's algorithm for mobile robot (turtlebot3_burger) in ROS.

* bagfiles are created by manually driving the robot with turtlebot3_teleop, topics recorded: '/scan' and '/odom'
* grid maps can be created from bagfiles using [create_from_rosbag.py](scripts/create_from_rosbag.py)
* grid maps can be created in real time using [rtime_gmapping_node.py](scripts/rtime_gmapping_node.py)

# Content
* [bagfiles](bagfiles) -> folder containing recorded rosbag files
* [maps](maps) -> folder containing images of Gazebo maps as well as output grid maps
* [papers](papers) -> materials used for the project 
* [scripts](scripts) -> python scripts 

# Results

![real_time_gmapping](https://user-images.githubusercontent.com/72970001/111978505-3e858580-8b04-11eb-9726-74e98ce94b16.gif)

![stage_4_compared](https://user-images.githubusercontent.com/72970001/111869094-eae92f80-897d-11eb-8ad8-7cfb21e23eaf.png)

![world_compared](https://user-images.githubusercontent.com/72970001/111869096-ecb2f300-897d-11eb-80fe-5737be27f72b.png)

![house_compared](https://user-images.githubusercontent.com/72970001/111869077-d9078c80-897d-11eb-8cb7-c6c33618d49a.png)
