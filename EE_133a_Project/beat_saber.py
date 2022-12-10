import rclpy
import numpy as np

from hw6code.GeneratorNode     import GeneratorNode
from hw6code.KinematicChain    import KinematicChain
from hw5code.TransformHelpers  import *

from rclpy.qos              import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import MarkerArray

from EE_133a_Project.cube_organizer import CubeOrganizer



## Convert cube_pos_type number to the center of cube position
pos_type_to_position = {
    1:  [-0.6, 0.75, 0.3],
    2:  [-0.2, 0.75, 0.3],
    3:  [0.2,  0.75, 0.3],
    4:  [0.6,  0.75, 0.3],
    5:  [-0.6, 0.75, 0.7],
    6:  [-0.2, 0.75, 0.7],
    7:  [0.2,  0.75, 0.7],
    8:  [0.6,  0.75, 0.7],
    9:  [-0.6, 0.75, 1.1],
    10: [-0.2, 0.75, 1.1],
    11: [0.2,  0.75, 1.1],
    12: [0.6,  0.75, 1.1]
}


def spline(t, T, p0, pf):
    p = p0 + (pf-p0) * (3*t**2/T**2 - 2*t**3/T**3)
    v =      (pf-p0) * (6*t   /T**2 - 6*t**2/T**3)
    return (p, v)


def compute_pre_post_slice_poses(cube_pos_type, cube_dir_type, cube_length):
    ## Decide pre_slice_pos and post_slice_pos based on cube_pos_type and cube_dir_type

    cube_pos = np.array(pos_type_to_position[cube_pos_type]).reshape(3, 1)

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


def compute_slice_perp_axis(pre_slice_pos, post_slice_pos):
    '''
    Computes the axis we allow the blade to rotate around for each slice. The plane
    of the slice intersects both the slice motion and the y-axis, so its normal is
    perpendicular to both of these
    '''
    slice_perp_axis = cross(post_slice_pos - pre_slice_pos, ey())
    slice_perp_axis /= np.linalg.norm(slice_perp_axis)
    
    return slice_perp_axis


def compute_inter_slice_w(slice1_perp_axis, slice2_perp_axis, cur_cycle_time, execution_time):
    '''
    Interpolate between perpendicular axis for each slice so the robot doesn't freak out
    when we suddenly switch
    '''
    
    # if the dot product is < 0, then the two axes are >90deg apart, so just flip
    # the second one to fix this
    if np.dot(slice1_perp_axis.T, slice2_perp_axis)[0,0] < 0:
    	slice2_perp_axis = -slice2_perp_axis
    	
    # find axis to rotate from one to the other, and angle to rotate through
    rot_axis = cross(slice1_perp_axis, slice2_perp_axis)
    
    # if they're parallel the cross product won't work and we should just stay at the same
    # value the whole time
    if np.linalg.norm(rot_axis) == 0:
        return slice1_perp_axis, np.zeros((3,1))
    
    rot_angle = np.arcsin(np.linalg.norm(rot_axis))
    rot_axis /= np.linalg.norm(rot_axis)
    
    # now we can spline between the two
    (sp, spdot) = spline(cur_cycle_time, execution_time, 0, 1)
    intermed_axis = Rote(rot_axis, sp * rot_angle) @ slice1_perp_axis
    wd = rot_angle * spdot * rot_axis
    
    return intermed_axis, wd



class Trajectory():
    # Initialization.
    def __init__(self, node):
        # Set up the kinematic chain object.
        self.chain = KinematicChain(node, 'world', 'tip-sliding-roll', self.jointnames())

        self.q0 = np.radians(np.array([0, 90, 0, -90, 0, 0, 0, 0, 0]).reshape((-1,1)))
        self.p0 = np.array([0.0, 0.55, 1.0]).reshape((-1,1))
        self.R0 = Reye()

        self.q = self.q0
        self.p = self.p0
        # the perpendicular axis for the previous slice, can init arbitrarily to z axis
        self.prev_slice_perp_axis = ez() 
        self.chain.setjoints(self.q)
        self.lam = 5
        self.lam2 = 10
        self.gamma = 0.1
        self.cumulative_qdot = np.zeros((9, 1))
        self.max_qdot = -1

        self.init = True               ## Tracking if have performed one slice or not

        self.slice_duration = 0.5      ## Time duration to perform a slice. This should depend on cube velocity
        self.inter_slice_duration = 1.5  ## Time duration to move from a post slice pos to the pre slice pos of the next cube  
        self.start_time = 0            ## Start time of targetting each cube. Set to 0 or the previous slice's end time
        self.cube_idx = 0              ## The index of the cube that we are going to slice next

        ## Info of all incoming cubes. Each cube is defined by a position type, direction type, and arrival time
        ## Arrival time is defined as the time of the cube arriving at the position that the robot should initiate a slice
        # self.cubes = np.array([         
        #     [1, 1, 2, 0],                  
        #     [2, 2, 4, 1],
        #     [3, 3, 6, 0], 
        #     [4, 4, 8, 1],
        #     [5, 5, 10, 0],
        #     [6, 6, 12, 1],
        #     [7, 7, 14, 0],
        #     [8, 8, 16, 1],
        #     [9, 1, 18, 0],
        #     [10, 2, 20, 1],
        #     [11, 3, 22, 0],
        #     [12, 4, 24, 1]])    


        # self.cubes = np.array([
        #     [1, 1, 5, 0],                  
        #     [2, 2, 5, 1],
        #     [3, 3, 5, 0], 
        #     [4, 4, 5, 1],
        #     [5, 5, 5, 0],
        #     [6, 6, 5, 1],
        #     [7, 7, 5, 0],
        #     [8, 8, 5, 1],
        #     [9, 1, 5, 0],
        #     [10, 2, 5, 1],
        #     [11, 3, 5, 0],
        #     [12, 4, 5, 1]])      

        self.cubes = np.array([         
            [1, 1, 11, 0],                  
            [2, 2, 12, 0],
            [3, 3, 13, 0], 
            [4, 4, 14, 0],
            [5, 5, 15, 0],
            [6, 6, 16, 0],
            [7, 7, 17, 0],
            [8, 8, 18, 0],
            [9, 1, 19, 0],
            [10, 2, 20, 0],
            [11, 3, 21, 0],
            [12, 4, 22, 0]])  

        self.last_post_slice_pos = np.zeros((3, 1))  ## The last finishing position of the robot

        ## For cube visualization
        self.v = -2
        self.l = 0.3
        self.cube_organizer = CubeOrganizer(self.cubes, self.v, self.l, node)

        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.pub = node.create_publisher(
            MarkerArray, '/visualization_marker_array', quality)

        # Create a timer to keep calculating/sending commands.
        self.timer = node.create_timer(1 / float(100), self.update_marker_array)
        self.dt    = self.timer.timer_period_ns * 1e-9
        self.t     = 0.0


    def update_marker_array(self):
        self.cube_organizer.update(self.dt)
        self.pub.publish(self.cube_organizer.get_marker_array())


    # Declare the joint names.
    def jointnames(self):
        return ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'blade-length', 'blade-roll']

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

        # print('t = ', t, 
        #     ' Current cube index = ', self.cube_idx, 
        #     ' position type = ', cube_pos_type, 
        #     ' direction type = ', cube_dir_type, 
        #     ' arrival time = ', cube_arrival_time)


        if not self.init:
            # self.p = self.last_post_slice_pos
            last_cube_arrival_time = self.cubes[self.cube_idx - 1, -1]
            self.start_time = last_cube_arrival_time + self.slice_duration


        ## Calculate pre slice and post slice positions based on the next cube's type
        pre_slice_pos, post_slice_pos = compute_pre_post_slice_poses(cube_pos_type, cube_dir_type, self.l)
        # self.last_post_slice_pos = post_slice_pos

        ## Decide whether we are in slicing motion or pre-slice motion
        ## If the next incoming cube has not arrived, we are in pre slice status
        if cube_arrival_time > t:

            print('Inter slice')
            print('\tself.start_time = ', self.start_time, ' pre_slice_pos = ', list(pre_slice_pos.reshape(-1)))

            if t > self.start_time + self.inter_slice_duration:
                pd = pre_slice_pos
                vd = np.zeros((3, 1))
                # keep axis steady at the value for this slice
                slice_perp_axis = compute_slice_perp_axis(pre_slice_pos, post_slice_pos)
                wd = np.zeros((3,1))
            else:
                pd, vd = compute_inter_slice_path(
                    self.p, 
                    pre_slice_pos, 
                    cur_cycle_time=t - self.start_time, 
                    execution_time=min(cube_arrival_time - self.start_time, self.inter_slice_duration))
                    
                # do interpolation of slice perp axis
                slice_perp_axis, wd = compute_inter_slice_w(
                    self.prev_slice_perp_axis, 
                    compute_slice_perp_axis(pre_slice_pos, post_slice_pos), 
                    t - self.start_time,
                    min(cube_arrival_time - self.start_time, self.inter_slice_duration))

        ## If the next incoming cube has arrived, we are in slicing status
        else:

            print('Slicing')
            pd, vd = compute_slice_path(
                pre_slice_pos, 
                post_slice_pos, 
                cur_cycle_time=t - cube_arrival_time, 
                execution_time=self.slice_duration)
            slice_perp_axis = compute_slice_perp_axis(pre_slice_pos, post_slice_pos)
            wd = np.zeros((3,1))
            self.p = post_slice_pos
            self.prev_slice_perp_axis = slice_perp_axis

        # print('pd = ', list(pd.reshape(-1)))

        # we want to make this vector the z axis of a new frame (frame "1" in following
        # comments)
        # the x axis can be ey since we know it's perpendicular
        # y = z cross x
        # this matrix is 0|R_1
        task_R_frame_01 = np.hstack((ey(), cross(slice_perp_axis, ey()), slice_perp_axis))
        # this one is 1|R_0 which we need to go from frame 0 to 1
        task_R_frame_10 = task_R_frame_01.T

        tip_z = self.chain.Rtip()[:,2:3]
        # if the dot product is < 0, we're more than 90deg off, which means it'll
        # be easier to push towards the negative of the perpendicular axis
        # (it doesn't actually matter since we just care that it's parallel)
        if np.dot(tip_z.T, slice_perp_axis)[0,0] < 0:
            slice_perp_axis = -slice_perp_axis
        # find rotation error with cross product of tip z and our new z axis
        Rerr = cross(tip_z, slice_perp_axis)

        # add it to desired w and then transform into new frame
        new_w = wd + self.lam * Rerr
        # only want the x and y components since we don't care about rotation 
        # around the new z which is the free axis
        new_w_1 = (task_R_frame_10 @ new_w)[:2,:]

        # transform Jw into the new frame and cut off the bottom row we don't care about
        Jw_1 = (task_R_frame_10 @ self.chain.Jw())[:2,:]

        #err = np.vstack((ep(pd, self.chain.ptip()), eR(Rd, self.chain.Rtip())))
        #xdot = np.vstack((vd, wd))
        perr = ep(pd, self.chain.ptip())
        J = np.vstack((self.chain.Jv(), Jw_1))

        # qdot = J_pinv @ (xdot + self.lam * err)

        
        #J_weighted_inv = np.linalg.inv(J.T @ J + self.gamma**2 * np.eye(9)) @ J.T
        # find qdot for primary velocity plus secondary task in null space
        #qdot = (J_weighted_inv @ np.vstack((vd + self.lam * perr, new_w_1))
        #    + (np.eye(9) - J_weighted_inv @ J) @ secondary_qdot)

        # weight limited joint more and more heavily until it remains inside limits
        weight = 1
        qdot = None
        while qdot is None or not(-0.5 < (self.q + dt*qdot)[7] < 0.5):
            if not (qdot is None):
                print(f"[DEBUG] bad slider position {(self.q + dt*qdot)[7]}, limiting joint with weight {weight}")
            ## Use weighted inverse to account for the case where target position is unreachable
            # weight joints so that we heavily prefer using the fake joints if possible 
            W = np.diag(1/np.array([256,256,128,128,128,128,128,weight,1]) ** 2)
            Winv = np.linalg.inv(W)
            J_weighted_inv = W @ J.T @ np.linalg.inv(J @ W @ J.T)
            
            u, s, vT = np.linalg.svd(J_weighted_inv, full_matrices = False)
            # apply gamma to s values to avoid singularity
            # since this is already an inverse, we have already inverted the s values,
            # so this formula is a little bit different but notice for small values it's ~s
            s = s/(1 + (s**2) * (self.gamma**2))
            J_weighted_inv = u @ np.diag(s) @ vT
        
            # create secondary task to push the robot back towards identity rotation
            # to make its movement a bit more reasonable
            secondary_w = self.lam2 * eR(Reye(), self.chain.Rtip())
            Jw = self.chain.Jw()
            # use the joint weighting here too so the secondary task will also eventually
            # fit blade slider motion in the limits
            secondary_qdot = W @ Jw.T @ np.linalg.inv(Jw @ W @ Jw.T) @ secondary_w
            # also push the blade slider towards the middle so we tend to slice using the middle of the blade
            secondary_qdot = secondary_qdot + self.lam2 * np.array([0,0,0,0,0,0,0,-self.q[7,0],0]).reshape((-1,1))
            
            # find qdot for primary velocity plus secondary task in null space
            qdot = (J_weighted_inv @ np.vstack((vd + self.lam * perr, new_w_1))
                + (np.eye(9) - J_weighted_inv @ J) @ secondary_qdot)
            #print(J @ W @ J_weighted_inv @ np.vstack((vd + self.lam * perr, new_w_1)))
            #print(np.vstack((vd + self.lam * perr, new_w_1)))
            
            weight *= 2


        self.q += dt * qdot
        self.chain.setjoints(self.q)

        self.cumulative_qdot += np.abs(qdot)
        self.max_qdot = max(self.max_qdot, np.max(np.abs(qdot[:7])))
        print(self.cumulative_qdot)
        print(np.mean(self.cumulative_qdot[:7]), ' ', self.max_qdot)

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
