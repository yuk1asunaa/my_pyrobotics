import math
import numpy as np
from scipy.interpolate import interp1d
import sys
import pathlib
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from utils.angle import angle_mod

# motion parameter
L = 1.0  # wheel base
ds = 0.1  # course distance
v = 10.0 / 3.6  # velocity [m/s]


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

def pi_2_pi(angle):
    return angle_mod(angle)
        
def update(state, v, delta, dt, L):
    state.v = v
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.yaw = pi_2_pi(state.yaw)

    return state       
        
def generate_trajectory(s, km, kf, k0):
    n = s / ds
    time = s / v  # [s]

    if isinstance(time, np.ndarray):
        time = time[0]
    if isinstance(km, np.ndarray):
        km = km[0]
    if isinstance(kf, np.ndarray):
        kf = kf[0]

    tk = np.array([0.0, time / 2.0, time])
    kk = np.array([k0, km, kf])
    t = np.arange(0.0, time, time / n)
    fkp = interp1d(tk, kk, kind="quadratic")
    kp = [fkp(ti) for ti in t]
    dt = float(time / n)

    state = State()
    x, y, yaw = [state.x], [state.y], [state.yaw]

    for ikp in kp:
        state = update(state, v, ikp, dt, L)
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)

    return x, y, yaw

def generate_last_state(s, km, kf, k0):
    n = s / ds
    time = s / v  # [s]

    if isinstance(n, np.ndarray):
        n = n.item()
    if isinstance(time, np.ndarray):
        time = time.item()
    if isinstance(km, np.ndarray):
        km = km.item()
    if isinstance(kf, np.ndarray):
        kf = kf.item()

    tk = np.array([0.0, time / 2.0, time])
    kk = np.array([k0, km, kf])
    t = np.arange(0.0, time, time / n)
    fkp = interp1d(tk, kk, kind="quadratic")
    kp = [fkp(ti) for ti in t]
    dt = time / n

    state = State()

    _ = [update(state, v, ikp, dt, L) for ikp in kp]

    return state.x, state.y, state.yaw



# optimization parameter
max_iter = 100
h: np.ndarray = np.array([0.5, 0.02, 0.02]).T  # parameter sampling distance
cost_th = 0.1

show_animation = True


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              fc=fc, ec=ec, head_width=width, head_length=width)
    plt.plot(x, y)
    plt.plot(0, 0)


def calc_diff(target, x, y, yaw):
    d = np.array([target.x - x[-1],
                  target.y - y[-1],
                  pi_2_pi(target.yaw - yaw[-1])])

    return d


def calc_j(target, p, h, k0):
    xp, yp, yawp = generate_last_state(
        p[0, 0] + h[0], p[1, 0], p[2, 0], k0)
    dp = calc_diff(target, [xp], [yp], [yawp])
    xn, yn, yawn = generate_last_state(
        p[0, 0] - h[0], p[1, 0], p[2, 0], k0)
    dn = calc_diff(target, [xn], [yn], [yawn])
    d1 = np.array((dp - dn) / (2.0 * h[0])).reshape(3, 1)

    xp, yp, yawp = generate_last_state(
        p[0, 0], p[1, 0] + h[1], p[2, 0], k0)
    dp = calc_diff(target, [xp], [yp], [yawp])
    xn, yn, yawn = generate_last_state(
        p[0, 0], p[1, 0] - h[1], p[2, 0], k0)
    dn = calc_diff(target, [xn], [yn], [yawn])
    d2 = np.array((dp - dn) / (2.0 * h[1])).reshape(3, 1)

    xp, yp, yawp = generate_last_state(
        p[0, 0], p[1, 0], p[2, 0] + h[2], k0)
    dp = calc_diff(target, [xp], [yp], [yawp])
    xn, yn, yawn = generate_last_state(
        p[0, 0], p[1, 0], p[2, 0] - h[2], k0)
    dn = calc_diff(target, [xn], [yn], [yawn])
    d3 = np.array((dp - dn) / (2.0 * h[2])).reshape(3, 1)

    J = np.hstack((d1, d2, d3))

    return J


def selection_learning_param(dp, p, k0, target):
    mincost = float("inf")
    mina = 1.0
    maxa = 2.0
    da = 0.5

    for a in np.arange(mina, maxa, da):
        tp = p + a * dp
        xc, yc, yawc = generate_last_state(
            tp[0], tp[1], tp[2], k0)
        dc = calc_diff(target, [xc], [yc], [yawc])
        cost = np.linalg.norm(dc)

        if cost <= mincost and a != 0.0:
            mina = a
            mincost = cost

    #  print(mincost, mina)
    #  input()

    return mina


def show_trajectory(target, xc, yc):  # pragma: no cover
    plt.clf()
    plot_arrow(target.x, target.y, target.yaw)
    plt.plot(xc, yc, "-r")
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.1)


def optimize_trajectory(target, k0, p):
    for i in range(max_iter):
        xc, yc, yawc = generate_trajectory(p[0, 0], p[1, 0], p[2, 0], k0)
        dc = np.array(calc_diff(target, xc, yc, yawc)).reshape(3, 1)

        cost = np.linalg.norm(dc)
        if cost <= cost_th:
            print("path is ok cost is:" + str(cost))
            break

        J = calc_j(target, p, h, k0)
        try:
            dp = - np.linalg.inv(J) @ dc
        except np.linalg.linalg.LinAlgError:
            print("cannot calc path LinAlgError")
            xc, yc, yawc, p = None, None, None, None
            break
        alpha = selection_learning_param(dp, p, k0, target)

        p += alpha * np.array(dp)
        #  print(p.T)

        if show_animation:
            show_trajectory(target, xc, yc)
    else:
        xc, yc, yawc, p = None, None, None, None
        print("cannot calc path")

    return xc, yc, yawc, p


def optimize_trajectory_demo():

    #  target = motion_model.State(x=5.0, y=2.0, yaw=np.deg2rad(00.0))
    target = State(x=5.0, y=2.0, yaw=np.deg2rad(90.0))
    k0 = 0.0

    init_p = np.array([6.0, 0.0, 0.0]).reshape(3, 1)

    x, y, yaw, p = optimize_trajectory(target, k0, init_p)

    if show_animation:
        show_trajectory(target, x, y)
        plot_arrow(target.x, target.y, target.yaw)
        plt.axis("equal")
        plt.grid(True)
        plt.show()


def main():
    print(__file__ + " start!!")
    optimize_trajectory_demo()


if __name__ == '__main__':
    main()