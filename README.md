# Autonomous Navigation for the Search and Resue Robot

## Problem Description

The robot is placed in a grid-based "warehouse"environment to pick up and return boxes by autonomous navigation and SLAM tasks. In the grid-based environment, the positions of both boxes and obstacles are unknown. The robot is equipped with a line-of-sight sensor to collect distance/angle sensory data to help its localization and navigation. 

## Environment Setup
Project is developed under Python 2.7.15, Numpy 1.4.15 and Matplotlib 2.2.2. The code is run by execute:
```
Python TestingEnv.py
```
## Result Demonstration
The robot firstly searches its surroundings to find the closest box and meanwhile build the local map. Based on the map, the robot navigates to the box and get it back to the start point. For the next rounds, the robot follows the similiar steps until all boxes are cleared in the warehouse environment. A brief demonstration GIF is given below. 

![alt text](https://github.com/ztliu62/RobotNavigation_graphSLAM/blob/master/RobotPath.gif)
