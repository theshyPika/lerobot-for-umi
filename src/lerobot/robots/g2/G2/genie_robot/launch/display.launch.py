#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command


def generate_launch_description():
    # Get the package share directory
    pkg_share = FindPackageShare('genie_robot')
    # Set the URDF file path
    urdf_file = PathJoinSubstitution([pkg_share, 'urdf', 'G2_t2_crs', 'model.urdf'])
    
    # Set the RViz config file path
    rviz_config_file = PathJoinSubstitution([pkg_share, 'rviz', 'default.rviz'])
    
    # Read the URDF file content using Command
    robot_description_content = Command(['cat ', urdf_file])
    
    # Robot state publisher node
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(robot_description_content, value_type=str),
            'publish_frequency': 30.0
        }]
    )
    
    # Joint state publisher GUI node
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen'
    )
    
    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file],
        parameters=[{
            'use_sim_time': False
        }]
    )
    
    # Create the launch description
    ld = LaunchDescription()
    
    # Add the nodes
    ld.add_action(robot_state_publisher_node)
    ld.add_action(joint_state_publisher_gui_node)
    ld.add_action(rviz_node)
    
    return ld 