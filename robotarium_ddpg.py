import numpy as np
import tensorflow as tf
import math
import sys
sys.path.append('c:\\Users\\Qian Luo\\Desktop\\network project\\robotarium_python_simulator-master')

import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.graph import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *


MAX_EPISODES = 60000
MAX_EP_STEPS = 36
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 800
BATCH_SIZE = 32


class DDPG(object):
    def __init__(self, name, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(name, self.S,)
        q = self._build_c(name, self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + 'Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + 'Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(name, self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(name, self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, name, s, reuse=None, custom_getter=None):
        trainable = True #if reuse is None else False
        with tf.variable_scope(name + 'Actor', reuse=tf.AUTO_REUSE, custom_getter=custom_getter):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, name, s, a, reuse=None, custom_getter=None):
        trainable = True #if reuse is None else False
        with tf.variable_scope(name + 'Critic', reuse=tf.AUTO_REUSE, custom_getter=custom_getter):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

agent1_ddpg = DDPG('agent1',1,6,math.pi)

agent2_ddpg = DDPG('agent2',1,6,math.pi)

agent3_ddpg = DDPG('agent3',1,6,math.pi)

# get observtion based on distances and angles
def get_observation(d1,a1,d2,a2,ds,da):
    o_n = np.zeros(6)
    o_n[0] = d1
    o_n[1] = a1
    o_n[2] = d2
    o_n[3] = a2
    o_n[4] = ds
    o_n[5] = da
    return o_n

#get rewards based on 
def get_reward(d12,d13,agent1_action):
    r_n = 0
    colide_r1 = 0
    goal_r1 = 0
    r_12 = 0
    r_13 = 0

    goal_r1 = 1000*(math.pi/2 - abs(agent1_action))

    #goal_r1 =20*math.cos(agent1_action)

    if d12 <= 0.13:
        r_12 = -10000

    if d13 <= 0.13:
        r_13 = -10000

    colide_r1 = r_12 + r_13
    r_n = colide_r1 + goal_r1
     
    return r_n

# Transform x,y cordinates to polar
def get_polar(x1,y1,x2,y2): 
    angle = 0
    d = 0
    dy = y2-y1
    dx = x2-x1
    d = np.sqrt((pow(dx,2)+pow(dy,2)))
    if dx==0 and dy>0:
        angle = math.pi/2
    if dx==0 and dy<0:
        angle = -math.pi/2
    if dy==0 and dx>=0:
        angle = 0
    if dy==0 and dx<0:
        angle = math.pi
    if dx>0 and dy>0:
       angle = math.atan(dy/dx)
    elif dx<0 and dy>0:
       angle = math.pi + math.atan(dy/dx)
    elif dx<0 and dy<0:
       angle = -math.pi + math.atan(dy/dx)
    elif dx>0 and dy<0:
       angle = math.atan(dy/dx)
    return d, angle

#get distances and angles between angents and goals
def get_d(x):
    d1,a1 = get_polar(x[0, 0],x[1,0],0,0.2)   #distances between agents and goals
    d2,a2 = get_polar(x[0, 1],x[1,1],-0.2,0)
    d3,a3 = get_polar(x[0, 2],x[1,2],0,-0.2)
    d12,a12 = get_polar(x[0, 0],x[1,0],x[0, 1],x[1,1])     #distances between agents 
    d13,a13 = get_polar(x[0, 0],x[1,0],x[0, 2],x[1,2])
    d21,a21 = get_polar(x[0, 1],x[1,1],x[0, 0],x[1,0])
    d23,a23 = get_polar(x[0, 1],x[1,1],x[0, 2],x[1,2])
    d31,a31 = get_polar(x[0, 2],x[1,2],x[0, 0],x[1,0])
    d32,a32 = get_polar(x[0, 2],x[1,2],x[0, 1],x[1,1])
    return d1,a1,d2,a2,d3,a3,d12,a12,d13,a13,d21,a21,d23,a23,d31,a31,d32,a32

var = 3  # control exploration

if __name__ == '__main__':

    control_gain = 20

    # Experiment constants
    iterations = 1000
    N = 3

    #Limit maximum linear speed of any robot
    magnitude_limit = 0.4

    #initialization
    r = robotarium.Robotarium(number_of_robots=N, show_figure=False, sim_in_real_time=True)
    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
    si_to_uni_dyn = create_si_to_uni_dynamics()
    x = r.get_poses()
    d1,a1,d2,a2,d3,a3,d12,a12,d13,a13,d21,a21,d23,a23,d31,a31,d32,a32 = get_d(x)
    dxi = np.zeros((2, N))
    o_n1 = np.zeros(6)
    o_n2 = np.zeros(6)
    o_n3 = np.zeros(6)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    ep_reward = 0

    for episode in range(60000):
        r = robotarium.Robotarium(number_of_robots=N, show_figure=False, sim_in_real_time=True)
        x = r.get_poses()
        d1,a1,d2,a2,d3,a3,d12,a12,d13,a13,d21,a21,d23,a23,d31,a31,d32,a32 = get_d(x)   

        #drive the agents fron initial position to [0,-0.25],[0.25,0] and [0,0.25]
        while np.linalg.norm(x[:2, 0] - [0,-0.25])>=0.01 or np.linalg.norm(x[:2, 1] - [0.25,0])>=0.01 or np.linalg.norm(x[:2, 2] - [0,0.25])>=0.01 or abs(x[2,0] - math.pi/2)>=0.1 or abs(x[2,1] - math.pi)>=0.1 or abs(x[2,2] + math.pi/2)>=0.1:
            if np.linalg.norm(x[:2, 0] - [0,-0.25])<0.01 and abs(x[2,0] - math.pi/2)<0.1:
                dxi[:, 0] = np.zeros(2)
            else:
                dxi[:, 0] = -control_gain * (x[:2, 0] - [0,-0.25])

            if np.linalg.norm(x[:2, 1] - [0.25,0])<0.01 and abs(x[2,1] - math.pi)<0.1:
                dxi[:, 1] = np.zeros(2)
            else:
                dxi[:, 1] = -control_gain * (x[:2, 1] - [0.25,0])

            if np.linalg.norm(x[:2, 2] - [0,0.25])<0.01 and abs(x[2,2] + math.pi/2)<0.1:
                dxi[:, 2] = np.zeros(2)
            else:
                dxi[:, 2] = -control_gain * (x[:2, 2] - [0,0.25])
            
            # Threshold control inputs
            norms = np.linalg.norm(dxi, 2, 0)
            idxs_to_normalize = (norms > magnitude_limit)
            dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]
            dxu = si_to_uni_dyn(dxi, x)
            # Set the velocities of the robots
            r.set_velocities(np.arange(N), dxu)
            r.step()
            x = r.get_poses()

        d1,a1,d2,a2,d3,a3,d12,a12,d13,a13,d21,a21,d23,a23,d31,a31,d32,a32 = get_d(x) 

        #run with max iterations 36
        for i in range(36):

            #episode ends when collide
            if d12 <= 0.13 or d13 <= 0.13 or d23 <= 0.13:
                break

            #get the velocity based on observation    
            o_n1 = get_observation(d12,a12,d13,a13,a1,(x[2,0]-math.pi/2))
            agent1_action = agent1_ddpg.choose_action(o_n1)
            agent1_action = np.clip(np.random.normal(agent1_action, var), -2, 2)    # add randomness to action selection for exploration
            dxi[0, 0] = control_gain*d1*np.cos(a1+agent1_action)
            dxi[1, 0] = control_gain*d1*np.sin(a1+agent1_action)

            o_n2 = get_observation(d21,a21,d23,a23,a2,(x[2,1]-math.pi))
            agent2_action = agent2_ddpg.choose_action(o_n2)
            agent2_action = np.clip(np.random.normal(agent2_action, var), -2, 2) 
            dxi[0, 1] = control_gain*d2*np.cos(a2+agent2_action)
            dxi[1, 1] = control_gain*d2*np.sin(a2+agent2_action)

            o_n3 = get_observation(d31,a31,d32,a32,a3,(x[2,2]+math.pi/2))
            agent3_action = agent3_ddpg.choose_action(o_n3)
            agent3_action = np.clip(np.random.normal(agent3_action, var), -2, 2) 
            dxi[0, 2] = control_gain*d3*np.cos(a3+agent3_action)
            dxi[1, 2] = control_gain*d3*np.sin(a3+agent3_action)

            # Threshold control inputs
            norms = np.linalg.norm(dxi, 2, 0)
            idxs_to_normalize = (norms > magnitude_limit)
            dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]
            # Transform the single-integrator dynamcis to unicycle dynamics
            dxu = si_to_uni_dyn(dxi, x)
            # Set the velocities of the robots
            r.set_velocities(np.arange(N), dxu)
            r.step()
            x = r.get_poses()
            # Threshold control inputs
            norms = np.linalg.norm(dxi, 2, 0)
            idxs_to_normalize = (norms > magnitude_limit)
            dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]
            # Transform the single-integrator dynamcis to unicycle dynamics
            dxu = si_to_uni_dyn(dxi, x)
            # Set the velocities of the robots
            r.set_velocities(np.arange(N), dxu)
            r.step()
            x = r.get_poses()
            d1,a1,d2,a2,d3,a3,d12,a12,d13,a13,d21,a21,d23,a23,d31,a31,d32,a32 = get_d(x)

            #get rewards and store the experience
            r_n1 = get_reward(d12,d13,agent1_action)
            o_n1_next = get_observation(d12,a12,d13,a13,a1,(x[2,0]-math.pi/2))
            agent1_ddpg.store_transition(o_n1, agent1_action, r_n1, o_n1_next)
            if agent1_ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                agent1_ddpg.learn()

            r_n2 = get_reward(d21,d23,agent2_action)
            o_n2_next = get_observation(d21,a21,d23,a23,a2,(x[2,1]-math.pi))
            agent2_ddpg.store_transition(o_n2, agent2_action, r_n2, o_n2_next)
            if agent2_ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                agent2_ddpg.learn()

            r_n3 = get_reward(d31,d32,agent3_action)
            o_n3_next = get_observation(d31,a31,d32,a32,a3,(x[2,2]+math.pi/2))
            agent3_ddpg.store_transition(o_n3, agent3_action, r_n3, o_n3_next)
            if agent3_ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                agent3_ddpg.learn()

            ep_reward += (r_n1 + r_n2 + r_n3)

        if episode % 1000 == 0:
            saver.save(sess, './three_ma_weight/' + str(episode) + '.cptk')
            print('Episode:', episode, ' Reward: %i' % int(ep_reward/1000))
            ep_reward = 0

    sess.close()