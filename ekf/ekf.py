import smbus
import socket
import math
import numpy as np
import time

# ================= UDP =================
udp_ip = "192.168.10.11"
udp_port = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ================= MPU =================
bus = smbus.SMBus(1)
MPU_ADDR = 0x68

# EKF state: [roll, pitch, yaw, bgx, bgy, bgz]
x = np.zeros((6,1))

P = np.eye(6) * 0.01
Q = np.eye(6) * 0.001
R = np.eye(2) * 0.05

# LPF accel
ax_f = ay_f = az_f = 0.0
alpha = 0.05

# ===== yaw drift mitigation =====
gz_bias = 0.0
MAX_YAW_RATE   = math.radians(120.0)
MAX_ROLL_RATE  = math.radians(300.0)
MAX_PITCH_RATE = math.radians(300.0)

# Initial reference
roll0 = pitch0 = yaw0 = 0.0
initialized = False
init_count = 0
INIT_SAMPLES = 50

# Warm-up
WARMUP_TIME = 2.0
start_time = time.time()


# ================= MPU functions =================
def mpu_init():
    bus.write_byte_data(MPU_ADDR, 0x6B, 0x00)
    bus.write_byte_data(MPU_ADDR, 0x1B, 0x00)
    bus.write_byte_data(MPU_ADDR, 0x1A, 0x04)
    time.sleep(0.2)


def read_raw(addr):
    high = bus.read_byte_data(MPU_ADDR, addr)
    low  = bus.read_byte_data(MPU_ADDR, addr + 1)
    val = (high << 8) | low
    if val >= 0x8000:
        val -= 65536
    return val


def read_mpu():
    gx = read_raw(0x43) / 131.0
    gy = read_raw(0x45) / 131.0
    gz = read_raw(0x47) / 131.0

    ax = read_raw(0x3B) / 16384.0
    ay = read_raw(0x3D) / 16384.0
    az = read_raw(0x3F) / 16384.0

    return gx, gy, gz, ax, ay, az


# ================= EKF =================
def ekf_update(gx, gy, gz, ax, ay, az, dt):
    global x, P, ax_f, ay_f, az_f

    # LPF accel
    ax_f = alpha * ax + (1 - alpha) * ax_f
    ay_f = alpha * ay + (1 - alpha) * ay_f
    az_f = alpha * az + (1 - alpha) * az_f

    roll, pitch, yaw = x[0,0], x[1,0], x[2,0]
    bgx, bgy, bgz = x[3,0], x[4,0], x[5,0]

    # Gyro (bias removed)
    wx = math.radians(gx - bgx)
    wy = math.radians(gy - bgy)
    wz = math.radians(gz - bgz - gz_bias)

    roll_dot  = wx + math.sin(roll)*math.tan(pitch)*wy + math.cos(roll)*math.tan(pitch)*wz
    pitch_dot = math.cos(roll)*wy - math.sin(roll)*wz
    yaw_dot   = (math.sin(roll)/math.cos(pitch))*wy + (math.cos(roll)/math.cos(pitch))*wz

    # ===== rate clamp =====
    roll_dot  = max(min(roll_dot,  MAX_ROLL_RATE),  -MAX_ROLL_RATE)
    pitch_dot = max(min(pitch_dot, MAX_PITCH_RATE), -MAX_PITCH_RATE)
    yaw_dot   = max(min(yaw_dot,   MAX_YAW_RATE),   -MAX_YAW_RATE)

    # State prediction
    x[0,0] += roll_dot * dt
    x[1,0] += pitch_dot * dt
    x[2,0] += yaw_dot * dt

    P[:] = P + Q

    # ===== accel-based correction =====
    roll_acc  = math.atan2(ay_f, az_f)
    pitch_acc = math.atan2(-ax_f, math.sqrt(ay_f**2 + az_f**2))

    z = np.array([[roll_acc], [pitch_acc]])
    h = np.array([[x[0,0]], [x[1,0]]])

    H = np.zeros((2,6))
    H[0,0] = 1
    H[1,1] = 1

    y = z - h
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    x[:] = x + K @ y
    P[:] = (np.eye(6) - K @ H) @ P

    # ===== gyro bias auto calibration (STATIC ONLY) =====
    acc_norm = math.sqrt(ax_f**2 + ay_f**2 + az_f**2)

    is_static = (
        abs(gx) < 1.0 and
        abs(gy) < 1.0 and
        abs(gz) < 1.0 and
        abs(acc_norm - 1.0) < 0.05
    )

    if is_static:
        bias_alpha = 0.001
        x[3,0] = (1 - bias_alpha) * x[3,0] + bias_alpha * gx
        x[4,0] = (1 - bias_alpha) * x[4,0] + bias_alpha * gy
        x[5,0] = (1 - bias_alpha) * x[5,0] + bias_alpha * gz

    return x[0,0], x[1,0], x[2,0]


# ================= MAIN =================
mpu_init()
print("Calibration...")

# LPF 초기화
gx, gy, gz, ax, ay, az = read_mpu()
ax_f, ay_f, az_f = ax, ay, az

prev_time = time.time()

while True:
    gx, gy, gz, ax, ay, az = read_mpu()

    now = time.time()
    dt = now - prev_time
    prev_time = now

    roll, pitch, yaw = ekf_update(gx, gy, gz, ax, ay, az, dt)

    # Warm-up
    if time.time() - start_time < WARMUP_TIME:
        continue

    # Initial reference & yaw bias
    if not initialized:
        roll0  += roll
        pitch0 += pitch
        yaw0   += yaw
        gz_bias += gz
        init_count += 1

        if init_count >= INIT_SAMPLES:
            roll0  /= INIT_SAMPLES
            pitch0 /= INIT_SAMPLES
            yaw0   /= INIT_SAMPLES
            gz_bias /= INIT_SAMPLES
            initialized = True
            print("Initial attitude fixed")
        continue

    # Relative angles
    roll_deg  = math.degrees(roll  - roll0)
    pitch_deg = math.degrees(pitch - pitch0)
    yaw_deg   = math.degrees(yaw   - yaw0)

    msg = f"{roll_deg:.2f}, {pitch_deg:.2f}, {yaw_deg:.2f}"
    print(msg)
    sock.sendto(msg.encode(), (udp_ip, udp_port))

    time.sleep(0.1)
