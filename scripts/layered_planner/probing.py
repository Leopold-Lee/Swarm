#!/usr/bin/env python

"""
Autonumous navigation of robots formation with Layered path-planner:
- global planner: RRT
- local planner: Artificial Potential Fields
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from tools import *
from rrt import *
from potential_fields import *
import time

# for 3D plots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def move_obstacles(obstacles, params):
    obstacles[-1] += np.array([0.008, -0.0009]) * params.drone_vel*1.5

class Params:
    def __init__(self):
        self.animate_rrt = 0 # show RRT construction, set 0 to reduce time of the RRT algorithm
        self.visualize = 1 # show robots movement
        self.maxiters = 500 # max number of samples to build the RRT
        self.goal_prob = 0.05 # with probability goal_prob, sample the goal
        self.minDistGoal = 0.25 # [m], min distance os samples from goal to add goal node to the RRT
        self.extension = 0.8 # [m], extension parameter: this controls how far the RRT extends in each step.
        self.world_bounds_x = [-2.5, 2.5] # [m], map size in X-direction
        self.world_bounds_y = [-2.5, 2.5] # [m], map size in Y-direction
        self.drone_vel = 4.0 # [m/s]
        self.ViconRate = 100 # [Hz]
        self.influence_radius = 0.15 # [m] potential fields radius, defining repulsive area size near the obstacle
        self.goal_tolerance = 0.05 # [m], maximum distance threshold to reach the goal
        self.num_robots = 4 # number of robots in the formation
        self.interrobots_dist = 0.3 # [m], distance between robots in default formation
        self.max_sp_dist = 0.2 * self.drone_vel# * np.sqrt(self.num_robots) # [m], maximum distance between current robot's pose and the sp from global planner

class AttackParams:
    def __init__(self):
        self.attack_target = 0
        self.victim_index_1 = 0
        self.victim_index_2 = 0
        self.victim_index_3 = 0
        self.spawntime = 20
        self.obst_size_bit = 0.1
        self.vel_atk_drone = 6

class Robot:
    def __init__(self, id):
        self.id = id
        self.sp = np.array([0, 0])
        self.sp_global = np.array([0,0])
        self.route = np.array([self.sp])
        self.vel_array = []
        self.U_a = 0 # attractive APF function
        self.U_r = 0 # repulsive APF function
        self.U = 0 # total APF function
        self.leader = False

    def local_planner(self, obstacles, params):
        """
        This function computes the next_point
        given current location (self.sp) and potential filed function, f.
        It also computes mean velocity, V, of the gradient map in current point.
        """
        obstacles_grid = grid_map(obstacles)
        self.U, self.U_a, self.U_r = combined_potential(obstacles_grid, self.sp_global, params.influence_radius)
        [gy, gx] = np.gradient(-self.U)
        iy, ix = np.array( meters2grid(self.sp), dtype=int )
        w = 20 # smoothing window size for gradient-velocity
        ax = np.mean(gx[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
        ay = np.mean(gy[ix-int(w/2) : ix+int(w/2), iy-int(w/2) : iy+int(w/2)])
        # ax = gx[ix, iy]; ay = gy[ix, iy]
        self.V = params.drone_vel * np.array([ax, ay])
        self.vel_array.append(norm(self.V))
        dt = 0.01 * params.drone_vel / norm([ax, ay]) if norm([ax, ay])!=0 else 0.01
        # self.sp += dt**2/2. * np.array( [ax, ay] )
        self.sp += dt*np.array( [ax, ay] ) #+ 0.1*dt**2/2. * np.array( [ax, ay] )
        self.route = np.vstack( [self.route, self.sp] )

def visualize2D():
    draw_map(obstacles)
    # draw_gradient(robots[1].U) if params.num_robots>1 else draw_gradient(robots[0].U)
    for robot in robots: plt.plot(robot.sp[0], robot.sp[1], '^', color='blue', markersize=10, zorder=15) # robots poses
    robots_poses = []
    for robot in robots: robots_poses.append(robot.sp)
    robots_poses.sort(key=lambda p: atan2(p[1]-centroid[1],p[0]-centroid[0]))
    plt.gca().add_patch( Polygon(robots_poses, color='yellow') )
    plt.plot(centroid[0], centroid[1], '*', color='b', markersize=10, label='Centroid position')
    plt.plot(robot1.route[:,0], robot1.route[:,1], linewidth=2, color='green', label="Leader's path", zorder=10)
    # for robot in robots[1:]: plt.plot(robot.route[:,0], robot.route[:,1], '--', linewidth=2, color='green', zorder=10)
    plt.plot(P[:,0], P[:,1], linewidth=3, color='orange', label='Global planner path')
    plt.plot(traj_global[sp_ind,0], traj_global[sp_ind,1], 'ro', color='blue', markersize=7, label='Global planner setpoint')
    plt.plot(xy_start[0],xy_start[1],'bo',color='red', markersize=20, label='start')
    plt.plot(xy_goal[0], xy_goal[1],'bo',color='green', markersize=20, label='goal')
    plt.legend()

# Initialization
init_fonts(small=12, medium=16, big=26)
params = Params()
attack_params = AttackParams()
xy_start = np.array([1.2, 1.25])
xy_goal =  np.array([2.2, -2.2])
# xy_goal =  np.array([1.3, 1.0])

# Obstacles map construction
obstacles = [
    # wall
    np.array([[-1.0, 0], [2.5, 0.], [2.5, 0.3], [-1.0, 0.3]]),
    np.array([[0.5, 2.0], [0.8, 2.0], [0.8, 2.5], [0.5, 2.5]]),
    np.array([[0.0, -2.5], [0.3, -2.5], [0.3, -1.6], [0.0, -1.6]]),
    np.array([[-1.3, 0.3], [-0.7, 0.3], [-0.7, 0.6], [-1.3, 0.6]]),
    np.array([[-1.0, 1.8], [-0.7, 1.8], [-0.7, 2.5], [-1.0, 2.5]]),
    np.array([[-2.0, -2.47], [-1.4, -2.47], [-1.4, -1.2], [-2.0, -1.2]]),
    # room
    np.array([[-2.5, -2.5], [2.5, -2.5], [2.5, -2.47], [-2.5, -2.47]]),
    np.array([[-2.5, 2.47], [2.5, 2.47], [2.5, 2.5], [-2.5, 2.5]]),
    np.array([[-2.5, -2.47], [-2.47, -2.47], [-2.47, 2.47], [-2.5, 2.47]]),
    np.array([[2.47, -2.47], [2.5, -2.47], [2.5, 2.47], [2.47, 2.47]]),
    # attack drones
    np.array([[999.0, 2.0], [999.1, 2.0], [999.1, 2.1], [999.0, 2.1]]),
    np.array([[999.0, 2.0], [999.1, 2.0], [999.1, 2.1], [999.0, 2.1]]),
    # moving obstacles
    np.array([[-2.6, 1.4], [-2.5, 1.4], [-2.5, 1.5], [-2.6, 1.5]]),
]
"""" Narrow passage """
# passage_width = 0.3
# passage_location = 0.0
# obstacles = [
#             # narrow passage
#               np.array([[-2.5, -0.5], [-passage_location-passage_width/2., -0.5], [-passage_location-passage_width/2., 0.5], [-2.5, 0.5]]),
#               np.array([[-passage_location+passage_width/2., -0.5], [2.5, -0.5], [2.5, 0.5], [-passage_location+passage_width/2., 0.5]]),
#             ]
# obstacles = []
robots = None
robot1 = None
followers_sp = None
traj_global = None
P = None

def swarm_init():
    global robots, robot1, obstacles, followers_sp, traj_global, P
    robots = []

    for i in range(params.num_robots):
        robots.append(Robot(i+1))
    robot1 = robots[0]; robot1.leader=True
    P_long = rrt_path(obstacles, xy_start, xy_goal, params)
    print('Path Shortenning...')
    P = ShortenPath(P_long, obstacles, smoothiters=50) # P = [[xN, yN], ..., [x1, y1], [x0, y0]]

    traj_global = waypts2setpts(P, params)
    P = np.vstack([P, xy_start])
    plt.plot(P[:,0], P[:,1], linewidth=3, color='orange', label='Global planner path')
    plt.pause(0.5)

    robot1.route = np.array([traj_global[0,:]])
    robot1.sp = robot1.route[-1,:]

    followers_sp = formation(params.num_robots, leader_des=robot1.sp, v=np.array([0,-1]), l=params.interrobots_dist)
    for i in range(len(followers_sp)):
        robots[i+1].sp = followers_sp[i]
        robots[i+1].route = np.array([followers_sp[i]])

def swarm_step(sp_ind):
    global robots, robot1, obstacles, followers_sp, traj_global
    # leader's setpoint from global planner
    robot1.sp_global = traj_global[sp_ind,:]
    # correct leader's pose with local planner
    robot1.local_planner(obstacles, params)

    """ adding following robots in the swarm """
    # formation poses from global planner
    followers_sp_global = formation(params.num_robots, robot1.sp_global, v=normalize(robot1.sp_global-robot1.sp), l=params.interrobots_dist)
    for i in range(len(followers_sp_global)): robots[i+1].sp_global = followers_sp_global[i]
    for p in range(len(followers_sp)): # formation poses correction with local planner
        # robots repel from each other inside the formation
        robots_obstacles_sp = [x for i,x in enumerate(followers_sp + [robot1.sp]) if i!=p] # all poses except the robot[p]
        robots_obstacles = poses2polygons( robots_obstacles_sp ) # each drone is defined as a small cube for inter-robots collision avoidance
        obstacles1 = np.array(obstacles + robots_obstacles) # combine exisiting obstacles on the map with other robots[for each i: i!=p] in formation
        # follower robot's position correction with local planner
        robots[p+1].local_planner(obstacles1, params)
        followers_sp[p] = robots[p+1].sp

def make_target_coor(goal, vel_atk_drone, current_atk_coor):
    """
    goal + current_atk_coor -> direction vector
    direction vector + vel_atk_drone -> target_vector
    target_vector + current_atk_coor -> target_coor
    for example, vel_atk_drone = *** 4.0 m/s ***
    """
    direction_vector = goal - current_atk_coor

    target_coor = current_atk_coor + 0.01 * \
        vel_atk_drone * (direction_vector / norm(direction_vector))

    return target_coor

def move_attack_drone(special_target_1):
    temp_target_1 = special_target_1
    temp_bit = 0.1  # for the size of obstacle: default

    obstacles[-3] = [[temp_target_1[0], temp_target_1[1]], [temp_target_1[0] + temp_bit, temp_target_1[1]],
               [temp_target_1[0] + temp_bit, temp_target_1[1] + temp_bit], [temp_target_1[0], temp_target_1[1] + temp_bit]]
    # obs[-3] += np.array([-0.007, 0.0]) * params.drone_vel / 2
    # obs[-2] += np.array([-0.007, 0.0]) * params.drone_vel / 2
    # obstacles[-2] = [[temp_target_2[0], temp_target_2[1]], [temp_target_2[0] + temp_bit, temp_target_2[1]],
    #            [temp_target_2[0] + temp_bit, temp_target_2[1] + temp_bit], [temp_target_2[0], temp_target_2[1] + temp_bit]]

def attack(strategy):
    """
    Strategy a: to x-axis, in front of robots[3] by 0.2 m
    Strategy b: to x-axis, back of robots[3] by - 0.1 m
    Strategy c: move to center point precisely between robots[1] and robots[3]
    Strategy d: set the target as leader to move swarm north by + 0.2m
    Strategy e(new): move around based on centroid
    """
    if strategy == 'a':
        ATTACK_DISTANCE = 0.3
        target = attack_params.attack_target
        goal_for_atk = np.copy(robots[target].sp)
        goal_for_atk[0] -= ATTACK_DISTANCE
    elif strategy == 'b':
        ATTACK_DISTANCE = 0.2
        target = attack_params.attack_target
        goal_for_atk = np.copy(robots[target].sp)
        goal_for_atk[0] += ATTACK_DISTANCE
    elif strategy == 'c':
        goal_for_atk = 0.5 * (robots[0].sp + robots[3].sp)

    special_target = make_target_coor(
        goal_for_atk, attack_params.vel_atk_drone, obstacles[-3][0])
    
    move_attack_drone(special_target)

# Layered Motion Planning: RRT (global) + Potential Field (local)
if __name__ == '__main__':
    fig2D = plt.figure(figsize=(10,10))
    draw_map(obstacles)
    plt.plot(xy_start[0],xy_start[1],'bo',color='red', markersize=20, label='start')
    plt.plot(xy_goal[0], xy_goal[1],'bo',color='green', markersize=20, label='goal')

    swarm_init()
    print('Start movement...')
    t0 = time.time(); t_array = []
    sp_ind = 0
    while True: # loop through all the setpoint from global planner trajectory, traj_global
        t_array.append( time.time() - t0 )
        # print("Current time [sec]: ", time.time() - t0)
        dist_to_goal = norm(robot1.sp - xy_goal)
        if dist_to_goal < params.goal_tolerance: # [m]
            print('Goal is reached')
            break
        
        move_obstacles(obstacles, params) # change poses of some obstacles on the map

        if sp_ind == attack_params.spawntime:
            attack_drone_position = np.array([-0.5, 0.7])
            print("0. attacker is spawned at: " +
                str(attack_drone_position[0])+", "+str(attack_drone_position[1]))

            move_attack_drone(attack_drone_position)

        swarm_step(sp_ind)
        attack('c')
        # centroid pose:
        centroid = 0
        for robot in robots: centroid += robot.sp / len(robots)
        # metrics.centroid_path = np.vstack([metrics.centroid_path, centroid])
        # visualization
        if params.visualize:
            plt.cla()
            visualize2D()        

            plt.draw()
            plt.pause(0.01)

        # update loop variable
        if sp_ind < traj_global.shape[0]-1 and norm(robot1.sp_global - centroid) < params.max_sp_dist: sp_ind += 1

# close windows if Enter-button is pressed
plt.draw()
plt.pause(0.1)
raw_input('Hit Enter to close')
plt.close('all')