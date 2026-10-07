import numpy as np


class KalmanFilter:

    def __init__(self, cx, cy, dt=1.0, q_var = 1.0, r_var = 5.0, p_var = 10.0):
        
        # Define state [x, y, vx, vy]
        self.x = np.array([
            [cx],
            [cy],
            [0.0],
            [0.0]
        ])

        #motion model or state transition matrix
        self.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt,],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        #measurement model or observation matrix
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        #state uncertainty
        self.P = np.eye(4) * p_var

        
        #process noise
        self.Q = np.eye(4) * q_var


        #measurement noise
        self.R = np.array([
            [r_var, 0.0],
            [0.0, r_var]
        ])


    def set_dt(self, dt):
        self.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt,],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])



    def predict(self):

        #predict nxt states
        self.x = self.F @ self.x

        #predict new uncertainty
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x
    
    def update(self, cx, cy):

        z = np.array([
            [cx],
            [cy]
        ])

        #innovation or residual
        y = z - self.H @ self.x



        #calculate kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        #update state
        self.x = self.x + K @ y 

        #reduce uncertainty
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.x







