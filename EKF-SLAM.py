import numpy as np
import matplotlib.pyplot as plt

import math
import pathlib as pl


DT = 0.1 #time_tick
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

# Initial covariance
Cx = np.array([0.5, 0.5, np.deg2rad(30)]) ** 2
# Simulation parameter
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2


def calc_input():
    v = 1.0 # m/s
    yaw_rate = 0.1 # rad/s
    u = np.array([v, yaw_rate]).T
    return u

def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                 [0, 1.0, 0],
                 [0, 0, 1.0]])
    B = np.array([DT * math.cos(x[2, 0]), 0],
                 [DT *math.sin(x[2, 0]), 0],
                 [0.0, DT])
    x = (F @ x) + (B @ u)
    
def calc_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n  

def jacobi_motion(x, u):
    Fx = np.hstack(np.eye(STATE_SIZE), np.zeros(LM_SIZE) * calc_n_lm(x))
    
    jF = np.array([[0.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0])],
                [0.0, 0.0, DT * u[0, 0] * math.cos(x[2, 0])],
                [0.0, 0.0, 0.0]], dtype=float)

    G = np.eye(len(x)) + Fx.T @ jF @ Fx

    return G, Fx

def get_landmark_position_from_state(x, ind):
    lm = x[STATE_SIZE + LM_SIZE * ind : STATE_SIZE + LM_SIZE * (ind + 1), :]
    return lm

def observation(xTrue, xd, u, RFID):
    xTrue = motion_model(xTrue, u)
    for i in range(len(RFID[:, 0])):
        dx = RFID[i, 0] - xTrue[: ,0]
        dy = RFID[i, 1] - xTrue[:, 1]
        d = math.hypot(dx, dy)
    
def ekf_slam(xEst, pEst, u, z):
    # predict
    G, Fx = jacobi_motion(xEst, u)
    xEst = motion_model(xEst, u)
    PEst = G.T @ PEst @ G + Fx.T @ Cx @ Fx
    initP = np.eye(2)
    # pdate
    
       
    
    
    
    
    
    
    
    
    
    
def main():
    
    print("start!")
    time = 0.0
        
    # RFID positions [x, y]
    RFID = np.array([[10.0, -2.0],
                 [15.0, 10.0],
                 [3.0,  15.0],
                 [-5.0, 20.0]])
    
    
    xEst = np.zeros((STATE_SIZE, 1))    # EKF estimate
    xTrue = np.zeros((STATE_SIZE, 1))   # true state
    PEst = np.eye(STATE_SIZE)           # Initial covariance
    xDR = np.zeros((STATE_SIZE, 1))     # dead reckoning

    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    
    
    
    
    
    
    
    time += DT
    u = calc_input()

    xEst, pEst = ekf_slam()
    x_state = xEst[0:STATE_SIZE]
    
        
        
if __name__ == '__main__':
    main()