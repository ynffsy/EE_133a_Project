"""beatsaber.launch.py

   Launch the beat saber project

   This starts:
   1) beatsaber.py
   2) RVIZ to look at it

"""

import os

from ament_index_python.packages import get_package_share_directory as pkgdir

from launch                            import LaunchDescription
from launch.actions                    import Shutdown
from launch_ros.actions                import Node


#
# Generate the Launch Description
#
def generate_launch_description():

    ######################################################################
    # LOCATE FILES

    # Define the package.
    package = 'EE_133a_Project'

    # Locate the RVIZ configuration file.
    rvizcfg = os.path.join(pkgdir(package), 'rviz/viewurdf.rviz')

    # Locate/load the robot's URDF file (XML).
    urdf = os.path.join(pkgdir(package), 'urdf/sevenDOF_8_stat_tgt.urdf')
    with open(urdf, 'r') as file:
        robot_description = file.read()


    ######################################################################
    # PREPARE THE LAUNCH ELEMENTS

    # Configure a node for the trajectory generator
    beat_saber_node = Node(
        name       = 'beat_saber',
        package    = 'EE_133a_Project',
        executable = 'beat_saber',
        output     = 'screen',
        on_exit    = Shutdown())

    # Configure a node for the robot state publisher
    robot_publisher_node = Node(
        name       = 'robot_state_publisher',
        package    = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output     = 'screen',
        parameters = [{'robot_description': robot_description}],
        on_exit    = Shutdown())

    # Configure a node for RVIZ
    node_rviz = Node(
        name       = 'rviz', 
        package    = 'rviz2',
        executable = 'rviz2',
        output     = 'screen',
        arguments  = ['-d', rvizcfg],
        on_exit    = Shutdown())



    ######################################################################
    # COMBINE THE ELEMENTS INTO ONE LIST
    
    # Return the description, built as a python list.
    return LaunchDescription([

        # Start the demo and RVIZ
        beat_saber_node,
        robot_publisher_node,
        node_rviz,
    ])
