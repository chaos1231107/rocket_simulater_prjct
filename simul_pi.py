import smbus
import socket
import time
import math

udp_ip = "192.168.10.10"
udp_port = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

bus = smbus.SMBus(1)
MPU_ADDR = 0x68

roll = pitch = yaw = 0.0
prev_time = time.time()
def mpu6050_init():
    # Wake up
    bus.write_byte_data(MPU_ADDR, 0x6B, 0x00)

    # Gyro ±250°/s 범위 설정
    bus.write_byte_data(MPU_ADDR, 0x1B, 0x00)

    # DLPF = 20Hz (노이즈 대폭 감소)
    bus.write_byte_data(MPU_ADDR, 0x1A, 0x04)

    time.sleep(0.2)
def read_word(addr):
    high = bus.read_byte_data(MPU_ADDR, addr)
    low = bus.read_byte_data(MPU_ADDR, addr+1)
    val = (high << 8) + low
    val -= ((65535 - val) + 1)
    if val > 360:
        val -= 360 
    return val

def get_gyro_angle():
    gx = read_word(0x43) / 131.0
    gy = read_word(0x45) / 131.0
    gz = read_word(0x47) / 131.0
    
    return gx, gy, gz

while True:
    gx, gy, gz = get_gyro_angle()
    
    now = time.time()
    dt = now - prev_time
    prev_time = now

    #roll, pitch, yaw
    roll += gx * dt
    pitch += gy * dt
    yaw += gz * dt
    #normalization
    roll %= 360
    pitch %= 360
    yaw %= 360

    msg = f"{roll}, {pitch}, {yaw}"
    print(msg)
    sock.sendto(msg.encode(), (udp_ip, udp_port))
    time.sleep(0.1)
