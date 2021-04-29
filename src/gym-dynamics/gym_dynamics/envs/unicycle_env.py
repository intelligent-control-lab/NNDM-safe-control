import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gym

class UnicycleEnv(gym.Env):
    def __init__(self):
        self.max_vr = 2
        self.max_ar = 4
        self.max_vt = np.pi
        self.dt = 0.1
        self.max_u =  np.array([self.max_ar, self.max_vt])
        self.render_initialized=False
        self.step_cnt = 0


    def rk4(self, s, u, dt):
        dot_s1 = self.dynamics(s, u)
        dot_s2 = self.dynamics(s + 0.5*dt*dot_s1, u)
        dot_s3 = self.dynamics(s + 0.5*dt*dot_s2, u)
        dot_s4 = self.dynamics(s + dt*dot_s3, u)
        dot_s = (dot_s1 + 2*dot_s2 + 2*dot_s3 + dot_s4)/6.0
        return dot_s
    
    def dynamics(self, s, u):
        x = s[0]
        y = s[1]
        v = s[2]
        theta = s[3]

        dot_x = v * np.cos(theta)
        dot_y = v * np.sin(theta)
        dot_v = u[0]
        dot_theta = u[1]

        dot_s = np.array([dot_x, dot_y, dot_v, dot_theta])
        return dot_s

    def step(self, u, dt):
        u = self.filt_action(u)
        
        dot_state = self.rk4(self.state, u, self.dt)
        # dot_state = self.dynamics(self.state, u)

        self.state = self.state + dot_state * self.dt

        self.state = self.filt_state(self.state)

        self.step_cnt += 1
        done = False
        if self.step_cnt > 1000:
            done = True
        # state, reward, done, info
        info={
            "dot_state":dot_state,
        }
        return np.squeeze(np.array(self.state)), 0, done, info

    def reset(self):
        min_state = np.array([-10, -10, -self.max_vr, -np.pi])
        max_state = np.array([10, 10, self.max_vr, np.pi])
        self.state = np.random.uniform(min_state, max_state)
        return self.state

    def sample_action(self):
        action = np.random.uniform(-self.max_u, self.max_u)
        return action

    def filt_action(self, u):
        u = np.minimum(u,  self.max_u)
        u = np.maximum(u, -self.max_u)
        return u

    def filt_state(self, x):
        while x[3] > 3*np.pi:
            x[3] = x[3] - 2 * np.pi
        while x[3] < -3*np.pi:
            x[3] = x[3] + 2 * np.pi
        return x
    
    def get_unicycle_plot(self):
        theta = self.state[3]
        ang = (-self.state[3] + np.pi/2) / np.pi * 180
        s = self.state[[0,1]]
        t = self.state[[0,1]] + np.hstack([np.cos(theta), np.sin(theta)])
        c = s
        s = s - (t-s)
        return np.hstack([s[0], t[0]]), np.hstack([s[1], t[1]])

    def render(self):
        if not self.render_initialized:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            plt.ion()
            plt.xlim([-20,20])
            plt.ylim([-20,20])
            x,y = self.get_unicycle_plot()
            self.unicycle_line, = self.ax.plot(x, y)
            self.render_initialized = True
        x,y = self.get_unicycle_plot()
        self.unicycle_line.set_xdata(x)
        self.unicycle_line.set_ydata(y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.0001)