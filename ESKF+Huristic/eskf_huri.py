import smbus
import socket
import numpy as np
import math
import time
from collections import deque

# ================= UDP =================
udp_ip = "192.168.10.11"
udp_port = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ================= MPU =================
bus = smbus.SMBus(1)
MPU_ADDR = 0x68

# ================= Quaternion Utils =================
def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_norm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1,0,0,0])

def quat_from_rotvec(rv):
    a = np.linalg.norm(rv)
    if a < 1e-9:
        return np.array([1,0,0,0])
    axis = rv / a
    return np.hstack([math.cos(a/2), axis*math.sin(a/2)])

def quat_to_rotmat(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])

def quat_to_euler(q):
    w,x,y,z = q
    roll  = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(np.clip(2*(w*y-z*x), -1, 1))
    yaw   = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return np.degrees([roll, pitch, yaw])

def skew(v):
    return np.array([
        [0,-v[2],v[1]],
        [v[2],0,-v[0]],
        [-v[1],v[0],0]
    ])

# ================= MPU IO =================
def mpu_init():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0x00)
    bus.write_byte_data(MPU_ADDR, 0x1B, 0x00)
    bus.write_byte_data(MPU_ADDR, 0x1C, 0x00)
    bus.write_byte_data(MPU_ADDR, 0x1A, 0x03)
    time.sleep(0.2)

def read_raw(addr):
    h = bus.read_byte_data(MPU_ADDR, addr)
    l = bus.read_byte_data(MPU_ADDR, addr+1)
    v = (h<<8)|l
    return v-65536 if v>=0x8000 else v

def read_mpu():
    gx = read_raw(0x43)/131.0
    gy = read_raw(0x45)/131.0
    gz = read_raw(0x47)/131.0
    ax = read_raw(0x3B)/16384.0
    ay = read_raw(0x3D)/16384.0
    az = read_raw(0x3F)/16384.0
    return gx,gy,gz,ax,ay,az

# ================= Heuristic Static Detector =================
class StaticDetector:
    def __init__(self, window=25):
        self.acc = deque(maxlen=window)
        self.gyro = deque(maxlen=window)

    def update(self, acc, gyro):
        self.acc.append(np.linalg.norm(acc))
        self.gyro.append(np.linalg.norm(gyro))

    def is_static(self):
        if len(self.acc) < self.acc.maxlen:
            return False
        return (
            abs(np.mean(self.acc)-1.0) < 0.03 and
            np.var(self.acc) < 0.002 and
            np.mean(self.gyro) < np.radians(0.5)
        )

# ================= Convergence Detector  =================
class ConvergenceSnapper:
    """
    일정 시간 동안 '0이 아닌 상수'로 수렴하면
    그 값을 0으로 갈아치움
    """
    def __init__(self, window=40, eps=0.8):
        self.buf = deque(maxlen=window)
        self.eps = eps

    def update(self, x):
        self.buf.append(x)

    def should_snap(self):
        if len(self.buf) < self.buf.maxlen:
            return False
        arr = np.array(self.buf)
        if abs(np.mean(arr)) < 0.3:      # 이미 거의 0이면 패스
            return False
        return np.max(np.abs(arr - np.mean(arr))) < self.eps

# ================= Quaternion ESKF =================
class AttitudeESKF:
    def __init__(self):
        self.q = np.array([1.0,0,0,0])
        self.bg = np.zeros(3)
        self.P = np.eye(6)*0.01
        self.Q = np.diag([5e-5]*3+[1e-7]*3)
        self.R = np.eye(3)*0.02
        self.q_ref = None

    def predict(self, gyro, dt):
        omega = gyro - self.bg
        self.q = quat_norm(quat_mul(self.q, quat_from_rotvec(omega*dt)))

        F = np.zeros((6,6))
        F[0:3,0:3] = -skew(omega)
        F[0:3,3:6] = -np.eye(3)

        Phi = np.eye(6)+F*dt
        self.P = Phi @ self.P @ Phi.T + self.Q*dt
        self.P = (self.P+self.P.T)/2

    def update(self, acc):
        m = np.linalg.norm(acc)
        if abs(m-1.0) > 0.2:
            return
        acc /= m

        g_b = quat_to_rotmat(self.q).T @ np.array([0,0,1])
        r = acc - g_b
        if np.linalg.norm(r) > 0.5:
            return

        H = np.zeros((3,6))
        H[:,0:3] = -skew(g_b)

        S = H @ self.P@H.T + self.R
        K = self.P @ H.T@ np.linalg.inv(S)
        dx = K@r

        I = np.eye(6)
        self.P = (I-K @ H )@ self.P@(I-K @ H).T + K @ self.R @ K.T
        self.P = (self.P+self.P.T)/2

        self.q = quat_norm(quat_mul(quat_from_rotvec(dx[0:3]), self.q))
        self.bg += dx[3:6]

    def set_reference(self):
        self.q_ref = self.q.copy()

    def reset_yaw_reference(self):
        q_rel = quat_mul(self.q, quat_conj(self.q_ref))
        w,_,_,z = q_rel
        yaw_only = quat_norm(np.array([w,0,0,z]))
        self.q_ref = quat_mul(yaw_only, self.q_ref)

    def get_relative_euler(self):
        return quat_to_euler(quat_mul(self.q, quat_conj(self.q_ref)))

# ================= Main =================
mpu_init()
eskf = AttitudeESKF()
static_detector = StaticDetector()
snap_r = ConvergenceSnapper()
snap_p = ConvergenceSnapper()

# gyro bias
samples=[]
for _ in range(100):
    gx,gy,gz,_,_,_ = read_mpu()
    samples.append([gx,gy,gz])
    time.sleep(0.01)
eskf.bg = np.radians(np.mean(samples,axis=0))

# settle
eskf.set_reference()
prev = time.time()

while True:
    gx,gy,gz,ax,ay,az = read_mpu()
    now = time.time()
    dt = now-prev
    prev = now

    gyro = np.radians([gx,gy,gz])
    acc = np.array([ax,ay,az])

    eskf.predict(gyro, dt)
    eskf.update(acc)

    static_detector.update(acc, gyro)
    if static_detector.is_static():
        eskf.reset_yaw_reference()

    roll,pitch,yaw = eskf.get_relative_euler()
#    yaw = 0.0

    snap_r.update(roll)
    snap_p.update(pitch)

    if snap_r.should_snap():
        roll = 0.0
    if snap_p.should_snap():
        pitch = 0.0

    msg = f"{roll:.2f},{pitch:.2f},{yaw:.2f}"
    print(msg)
    sock.sendto(msg.encode(), (udp_ip,udp_port))
    time.sleep(0.05)
