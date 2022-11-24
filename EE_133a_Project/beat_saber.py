import rclpy
import numpy as np

from hw6code.GeneratorNode     import GeneratorNode
from hw6code.KinematicChain    import KinematicChain
from hw5code.TransformHelpers  import *



def spline(t, T, p0, pf):
    p = p0 + (pf-p0) * (3*t**2/T**2 - 2*t**3/T**3)
    v =      (pf-p0) * (6*t   /T**2 - 6*t**2/T**3)
    return (p, v)


def compute_pre_post_slice_poses(cube_pos_type, cube_dir_type):
    ## Decide pre_slice_pos and post_slice_pos based on cube_pos_type and cube_dir_type

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

    cube_pos = np.array(pos_type_to_position[cube_pos_type]).reshape(3, 1)
    cube_length = 0.25

    ## Convert cube_dir_type to the relative start position of a slice
    dir_type_to_slice_start_pos = {
        1: [0, 0, cube_length],
        2: [0, 0, -cube_length],
        3: [cube_length, 0, 0],
        4: [-cube_length, 0, 0],
        5: [cube_length, 0, cube_length],
        6: [-cube_length, 0, -cube_length],
        7: [-cube_length, 0, cube_length],
        8: [cube_length, 0, -cube_length]
    }

    slice_start_pos = np.array(dir_type_to_slice_start_pos[cube_dir_type]).reshape(3, 1)
    pre_slice_pos = cube_pos + slice_start_pos
    post_slice_pos = cube_pos - slice_start_pos

    return pre_slice_pos, post_slice_pos


def compute_slice_path(pre_slice_pos, post_slice_pos, cur_cycle_time, execution_time):

    (sp, spdot) = spline(cur_cycle_time, execution_time, 0, 1)
    pd = pre_slice_pos + (post_slice_pos - pre_slice_pos) * sp
    vd =                 (post_slice_pos - pre_slice_pos) * spdot

    return pd, vd


def compute_inter_slice_path(start_pos, pre_slice_pos, cur_cycle_time, execution_time):

    (sp, spdot) = spline(cur_cycle_time, execution_time, 0, 1)
    pd = start_pos + (pre_slice_pos - start_pos) * sp
    vd =             (pre_slice_pos - start_pos) * spdot

    return pd, vd



class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain = KinematicChain(node, 'world', 'tip', self.jointnames())

        self.q0 = np.radians(np.array([0, 90, 0, -90, 0, 0, 0]).reshape((-1,1)))
        self.p0 = np.array([0.0, 0.55, 1.0]).reshape((-1,1))
        self.R0 = Reye()

        self.q = self.q0
        self.p = self.p0
        self.chain.setjoints(self.q)
        self.lam = 10
        self.gamma = 0.1

        self.init = True               ## Tracking if have performed one slice or not

        self.slice_duration = 1        ## Time duration to perform a slice. This should depend on cube velocity
        self.inter_slice_duration = 3  ## Time duration to move from a post slice pos to the pre slice pos of the next cube  
        self.start_time = 0            ## Start time of targetting each cube. Set to 0 or the previous slice's end time
        self.cube_idx = 0              ## The index of the cube that we are going to slice next

        ## Info of all incoming cubes. Each cube is defined by a position type, direction type, and arrival time
        ## Arrival time is defined as the time of the cube arriving at the position that the robot should initiate a slice
        self.cubes = np.array([         
            [6, 2, 3],                  
            [3, 1, 8],
            [5, 8, 10], 
            [4, 7, 15],
            [1, 5, 18],
            [8, 4, 20]])            

        self.last_post_slice_pos = np.zeros((3, 1))  ## The last finishing position of the robot


    # Declare the joint names.
    def jointnames(self):
        return ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7']

    # Evaluate at the given time.  This was last called (dt) ago.
    def evaluate(self, t, dt):

        ## Get information on the current incoming cube
        cube_cur = self.cubes[self.cube_idx]
        cube_arrival_time = cube_cur[2]

        ## While the current time is larger than cube arrival time + slice execution time, move onto the next cube if there is any
        while t > cube_arrival_time + self.slice_duration:
            self.cube_idx += 1
            self.init = False

            ## If all cubes have already passed, stop the program
            if self.cube_idx >= self.cubes.shape[0]:
                return None

            cube_cur = self.cubes[self.cube_idx]
            cube_arrival_time = cube_cur[2]

        cube_pos_type = cube_cur[0]
        cube_dir_type = cube_cur[1]

        print('t = ', t, 
            ' Current cube index = ', self.cube_idx, 
            ' position type = ', cube_pos_type, 
            ' direction type = ', cube_dir_type, 
            ' arrival time = ', cube_arrival_time)


        if not self.init:
            # self.p = self.last_post_slice_pos
            last_cube_arrival_time = self.cubes[self.cube_idx - 1, -1]
            self.start_time = last_cube_arrival_time + self.slice_duration


        ## Calculate pre slice and post slice positions based on the next cube's type
        pre_slice_pos, post_slice_pos = compute_pre_post_slice_poses(cube_pos_type, cube_dir_type)
        # self.last_post_slice_pos = post_slice_pos

        ## Decide whether we are in slicing motion or pre-slice motion
        ## If the next incoming cube has not arrived, we are in pre slice status
        if cube_arrival_time > t:

            print('Inter slice')
            print('\tself.start_time = ', self.start_time, ' pre_slice_pos = ', list(pre_slice_pos.reshape(-1)))

            if t > self.start_time + self.inter_slice_duration:
                pd = pre_slice_pos
                vd = np.zeros((3, 1))
            else:
                pd, vd = compute_inter_slice_path(
                    self.p, 
                    pre_slice_pos, 
                    cur_cycle_time=t - self.start_time, 
                    execution_time=min(cube_arrival_time - self.start_time, self.inter_slice_duration))

        ## If the next incoming cube has arrived, we are in slicing status
        else:

            print('Slicing')
            pd, vd = compute_slice_path(
                pre_slice_pos, 
                post_slice_pos, 
                cur_cycle_time=t - cube_arrival_time, 
                execution_time=self.slice_duration)
            self.p = post_slice_pos


        print('pd = ', list(pd.reshape(-1)))

        Rd = Reye()
        wd = np.zeros((3,1))


        err = np.vstack((ep(pd, self.chain.ptip()), eR(Rd, self.chain.Rtip())))
        xdot = np.vstack((vd, wd))
        J = np.vstack((self.chain.Jv(), self.chain.Jw()))
        J_pinv = np.linalg.pinv(J)

        # qdot = J_pinv @ (xdot + self.lam * err)

        ## Use weighted inverse to account for the case where target position is unreachable
        J_weighted_inv = np.linalg.inv(J.T @ J + self.gamma ** 2 * np.eye(7)) @ J.T
        qdot = J_weighted_inv @ (xdot + self.lam * err) 

        self.q += dt * qdot
        self.chain.setjoints(self.q)

        return (self.q.flatten().tolist(), qdot.flatten().tolist())



def main(args=None):
    # Initialize ROS and the generator node (100Hz) for the Trajectory.
    rclpy.init(args=args)
    generator = GeneratorNode('generator', 100, Trajectory)

    # Spin, until interrupted or the trajectory ends.
    generator.spin()

    # Shutdown the node and ROS.
    generator.shutdown()
    rclpy.shutdown()



if __name__ == "__main__":
    main()