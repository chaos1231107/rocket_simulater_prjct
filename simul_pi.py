import smbus
import socket
import time
import math

# ---------- UDP Setup ----------
UDP_IP = "192.168.10.10"   # 맥북 또는 PC IP 주소
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ---------- I2C Setup ----------
bus = smbus.SMBus(1)
MPU_ADDR = 0x68

# ---------- MPU6050 초기 설정 ----------
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
    low = bus.read_byte_data(MPU_ADDR, addr + 1)
    val = (high << 8) + low
    if val >= 0x8000:
        val = -((65535 - val) + 1)
    return val

# ---------- Gyro Offsets ----------
gx_offset = 0
gy_offset = 0
gz_offset = 0

def calibrate_gyro(samples=500):
    global gx_offset, gy_offset, gz_offset
    print("Gyro Offset Calibration 중... 센서를 가만히 두세요.")

    gx_sum = 0
    gy_sum = 0
    gz_sum = 0

    for i in range(samples):
        gx = read_word(0x43) / 131.0
        gy = read_word(0x45) / 131.0
        gz = read_word(0x47) / 131.0

        gx_sum += gx
        gy_sum += gy
        gz_sum += gz

        time.sleep(0.002)

    gx_offset = gx_sum / samples
    gy_offset = gy_sum / samples
    gz_offset = gz_sum / samples

    print(f"Calibration 완료 → gx_offset={gx_offset:.3f}, gy_offset={gy_offset:.3f}, gz_offset={gz_offset:.3f}")


def get_gyro():
    gx = read_word(0x43) / 131.0
    gy = read_word(0x45) / 131.0
    gz = read_word(0x47) / 131.0

    # 오프셋 제거
    gx -= gx_offset
    gy -= gy_offset
    gz -= gz_offset

    return gx, gy, gz


# ---------- MAIN ----------
mpu6050_init()
calibrate_gyro()

while True:
    gx, gy, gz = get_gyro()

    msg = f"{gx:.4f},{gy:.4f},{gz:.4f}"
    print(msg)
    sock.sendto(msg.encode(), (UDP_IP, UDP_PORT))

    time.sleep(0.1)  # 100Hz

