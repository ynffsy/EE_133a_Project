import numpy as np

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray



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



class CubeOrganizer():

    def __init__(self, cube_info, v, l):
        ## v: y velocity of cubes (same for all cubes)
        ## l: size length of cubes (same for all cubes)

        self.n_cubes       = cube_info.shape[0]
        self.pos_types     = cube_info[:, 0:1]
        self.dir_types     = cube_info[:, 1:2]
        self.arrival_times = cube_info[:, 2:3]

        self.v = v
        self.l = l

        ## Compute all initial positions and initialize markers
        self.p0s = np.zeros((self.n_cubes, 3))
        self.markers = []

        for i in range(self.n_cubes):
            slice_pos = pos_type_to_position[self.pos_types[i, 0]]

            ## x and z positions stay constant
            x0 = slice_pos[0]
            z0 = slice_pos[2]
            y0 = slice_pos[1] - self.v * self.arrival_times[i]

            ## Given arrival times and slice position, we can calculate the initial y position
            self.p0s[i, 0] = x0
            self.p0s[i, 1] = y0
            self.p0s[i, 2] = z0

            self.markers.append(self.cube_marker(x0, y0, z0, l))            

        


    def update(self, dt):

        for i in range(self.n_cubes):
            self.markers[i].pose.position.y += self.v * dt


    def get_markers(self):
        return self.markers


    def cube_marker(self, x, y, z, l):

        print(x, ' ', y, ' ', z, ' ', l)
        # Create the cube marker.
        marker = Marker()
        marker.type               = Marker.CUBE

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x    = float(x)
        marker.pose.position.y    = float(y)
        marker.pose.position.z    = float(z)

        marker.scale.x            = float(l)
        marker.scale.y            = float(l)
        marker.scale.z            = float(l)

        marker.color.r            = 1.0
        marker.color.g            = 1.0
        marker.color.b            = 1.0
        marker.color.a            = 1.0     # Transparency

        return marker


