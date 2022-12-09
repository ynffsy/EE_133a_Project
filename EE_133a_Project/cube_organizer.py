import numpy as np

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point



## Convert cube_pos_type number to the center of cube position
pos_type_to_position = {
        1: [-0.75, 0.75, 0.25],
        2: [-0.25, 0.75, 0.25],
        3: [0.25,  0.75, 0.25],
        4: [0.75,  0.75, 0.25],
        5: [-0.75, 0.75, 0.75],
        6: [-0.25, 0.75, 0.75],
        7: [0.25,  0.75, 0.75],
        8: [0.75,  0.75, 0.75]
    }


## Convert cube_dir_type number to triangle position offsets
dir_type_to_offset = {
    1: np.array([[0, 0, 0], [1, 0, 1], [-1, 0, 1]]),
    2: np.array([[0, 0, 0], [-1, 0, -1], [1, 0, -1]]),
    3: np.array([[0, 0, 0], [1, 0, -1], [1, 0, 1]]),
    4: np.array([[0, 0, 0], [-1, 0, 1], [-1, 0, -1]]),
    5: np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]]),
    6: np.array([[0, 0, 0], [-1, 0, 0], [0, 0, -1]]),
    7: np.array([[0, 0, 0], [0, 0, 1], [-1, 0, 0]]),
    8: np.array([[0, 0, 0], [0, 0, -1], [1, 0, 0]]),
}



class CubeOrganizer():

    def __init__(self, cube_info, v, l, node):
        ## v: y velocity of cubes (same for all cubes)
        ## l: size length of cubes (same for all cubes)

        self.n_cubes       = cube_info.shape[0]
        self.pos_types     = cube_info[:, 0:1]
        self.dir_types     = cube_info[:, 1:2]
        self.arrival_times = cube_info[:, 2:3]
        self.blade_types   = cube_info[:, 3:4]

        self.v = v
        self.l = l
        self.node = node

        ## Compute all initial positions and initialize markers
        self.p0s = np.zeros((self.n_cubes, 3))
        self.cube_markers = []
        self.triangle_markers = []

        for i in range(self.n_cubes):
            slice_pos = pos_type_to_position[self.pos_types[i, 0]]
            dir_type = self.dir_types[i, 0]
            blade_type = self.blade_types[i, 0]

            ## x and z positions stay constant
            x0 = slice_pos[0]
            z0 = slice_pos[2]
            y0 = slice_pos[1] - self.v * self.arrival_times[i]

            ## Given arrival times and slice position, we can calculate the initial y position
            self.p0s[i, 0] = x0
            self.p0s[i, 1] = y0
            self.p0s[i, 2] = z0

            self.cube_markers.append(self.cube_marker(x0, y0, z0, l, dir_type, blade_type))
            self.triangle_markers.append(self.triangle_maker(x0, y0, z0, l, dir_type))       

        timestamp = node.get_clock().now().to_msg()

        for (i, marker) in enumerate(self.cube_markers):
            marker.header.stamp       = timestamp
            marker.header.frame_id    = 'world'
            marker.ns                 = 'cube'
            marker.action             = Marker.ADD
            marker.id                 = i 

        for (i, marker) in enumerate(self.triangle_markers):
            marker.header.stamp       = timestamp
            marker.header.frame_id    = 'world'
            marker.ns                 = 'triangle'
            marker.action             = Marker.ADD
            marker.id                 = i + self.n_cubes        

        self.marker_array = MarkerArray()
        self.marker_array.markers = self.cube_markers + self.triangle_markers


    def update(self, dt):

        for i in range(self.n_cubes):
            self.cube_markers[i].pose.position.y += self.v * dt

        for i in range(self.n_cubes):
            for j in range(3):
                self.triangle_markers[i].points[j].y += self.v * dt


    def get_marker_array(self):
        return self.marker_array


    def cube_marker(self, x, y, z, l, dir_type, blade_type):

        print(x, ' ', y, ' ', z, ' ', l)
        # Create the cube marker.
        marker = Marker()
        marker.type               = Marker.CUBE

        ## Rotate the cube by 45 degrees about the y axis if slice direction is slanted (dir type >= 5)
        if dir_type < 5:
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 0.0

        else:
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.3826834
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 0.9238795

        marker.pose.position.x    = float(x)
        marker.pose.position.y    = float(y)
        marker.pose.position.z    = float(z)

        marker.scale.x            = float(l)
        marker.scale.y            = float(l)
        marker.scale.z            = float(l)

        if blade_type == 0:  ## Blue
            marker.color.r            = 0.0
            marker.color.g            = 0.0
            marker.color.b            = 1.0
            marker.color.a            = 1.0     # Transparency

        elif blade_type == 1:  # Red
            marker.color.r            = 1.0
            marker.color.g            = 0.0
            marker.color.b            = 0.0
            marker.color.a            = 1.0     # Transparency

        else:
            marker.color.r            = 0.8
            marker.color.g            = 0.8
            marker.color.b            = 0.8
            marker.color.a            = 1.0     # Transparency

        return marker


    def triangle_maker(self, x, y, z, l, dir_type):

        marker = Marker()
        marker.type               = Marker.TRIANGLE_LIST

        surface_offset = l / 100  ## To avoid overlaping with the cube's surface
        proportion = 0.4  ## Proportion of triangle size and cube size

        ## Each point is a row. x, y, z are columns
        offset_template = dir_type_to_offset[dir_type]
    
        triangle_pos = np.zeros((3, 3))
        triangle_pos[:, 1:2] = y - l / 2 - surface_offset

        if dir_type < 5:
            triangle_pos[:, 0:1] = offset_template[:, 0:1] * l * proportion + x
            triangle_pos[:, 2:3] = offset_template[:, 2:3] * l * proportion + z
        else:
            triangle_pos[:, 0:1] = offset_template[:, 0:1] * l * proportion * 2 / np.sqrt(2) + x
            triangle_pos[:, 2:3] = offset_template[:, 2:3] * l * proportion * 2 / np.sqrt(2) + z

        p1 = Point()
        p1.x = float(triangle_pos[0, 0])
        p1.y = float(triangle_pos[0, 1])
        p1.z = float(triangle_pos[0, 2])

        p2 = Point()
        p2.x = float(triangle_pos[1, 0])
        p2.y = float(triangle_pos[1, 1])
        p2.z = float(triangle_pos[1, 2])

        p3 = Point()
        p3.x = float(triangle_pos[2, 0])
        p3.y = float(triangle_pos[2, 1])
        p3.z = float(triangle_pos[2, 2])

        marker.points = [p1, p2, p3]

        marker.scale.x            = float(1)
        marker.scale.y            = float(1)
        marker.scale.z            = float(1)

        marker.color.r            = 1.0
        marker.color.g            = 1.0
        marker.color.b            = 1.0
        marker.color.a            = 1.0

        return marker
