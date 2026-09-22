import ustruct
import time
from machine import I2C, Pin

class PCA9685:
    def __init__(self, i2c, address=0x40):
        self.i2c = i2c
        self.address = address
        self.reset()

    def _write(self, address, value):
        self.i2c.writeto_mem(self.address, address, bytearray([value]))

    def _read(self, address):
        return self.i2c.readfrom_mem(self.address, address, 1)[0]

    def reset(self):
        self._write(0x00, 0x00) # Mode1

    def freq(self, freq=None):
        if freq is None:
            return int(25000000.0 / 4096 / (self._read(0xfe) - 0.5))
        prescale = int(25000000.0 / 4096.0 / freq + 0.5)
        old_mode = self._read(0x00) # Mode 1
        self._write(0x00, (old_mode & 0x7F) | 0x10) # Mode 1, sleep
        self._write(0xfe, prescale) # Prescale
        self._write(0x00, old_mode) # Mode 1
        time.sleep_us(5)
        self._write(0x00, old_mode | 0xa1) # Mode 1, autoincrement on

    def pwm(self, index, on=None, off=None):
        if on is None or off is None:
            data = self.i2c.readfrom_mem(self.address, 0x06 + 4 * index, 4)
            return ustruct.unpack('<HH', data)
        data = ustruct.pack('<HH', on, off)
        self.i2c.writeto_mem(self.address, 0x06 + 4 * index,  data)

    def duty(self, index, value=None, invert=False):
        if value is None:
            pwm = self.pwm(index)
            if pwm == (0, 4096):
                value = 0
            elif pwm == (4096, 0):
                value = 4095
            value = pwm[1]
            if invert:
                value = 4095 - value
            return value
        if not 0 <= value <= 4095:
            raise ValueError("Out of range")
        if invert:
            value = 4095 - value
        if value == 0:
            self.pwm(index, 0, 4096)
        elif value == 4095:
            self.pwm(index, 4096, 0)
        else:
            self.pwm(index, 0, value)


class Servos:
    def __init__(self, i2c, address=0x40, freq=50, min_us=600, max_us=2400,
                 degrees=180):
        self.period = 1000000 / freq
        self.min_duty = self._us2duty(min_us)
        self.max_duty = self._us2duty(max_us)
        self.degrees = degrees
        self.freq = freq
        self.pca9685 = PCA9685(i2c, address)
        self.pca9685.freq(freq)

    def _us2duty(self, value):
        return int(4095 * value / self.period)

    def position(self, index, degrees=None, radians=None, us=None, duty=None):
        span = self.max_duty - self.min_duty
        if degrees is not None:
            duty = self.min_duty + span * degrees / self.degrees
        elif radians is not None:
            duty = self.min_duty + span * radians / math.radians(self.degrees)
        elif us is not None:
            duty = self._us2duty(us)
        elif duty is not None:
            pass
        else:
            return self.pca9685.duty(index)
        duty = min(self.max_duty, max(self.min_duty, int(duty)))
        self.pca9685.duty(index, duty)

    def release(self, index):
        self.pca9685.duty(index, 0)

i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=100000)
print(i2c.scan())

# min_us/max_us confirmed via pulse.py bring-up testing as the DS3218's
# safe, non-stalling pulse range (was 600-2400, tuned for the previous
# servos) - this is the range the normal "<ch>,<deg>\r" path below maps
# degrees 0-180 across.
ss = Servos(i2c, freq=60, min_us=500, max_us=2500)

# Raw pulse-width test mode (below) allows a bit more headroom than the
# confirmed-safe 500-2500us range on the upper end only, to probe for extra
# travel there. The lower end stays at the confirmed-safe 500us floor.
RAW_PULSE_MIN_US = 500.0
RAW_PULSE_MAX_US = 2600.0

deg = 0
while True:
    s = input()
    v = s.split(',')

    if len(v) == 3 and v[2] == 'u':
        # raw pulse-width test mode: "<channel>,<microseconds>,u\r"
        # Bypasses angle calibration and the Servos class's min_us/max_us
        # clamp entirely (drives the PCA9685 duty register directly), so
        # it can sweep beyond the confirmed-safe range when
        # bring-up-testing a newly-mounted servo before angle.py has
        # calibrated it. Used by pulse.py.
        chn = int(v[0])
        us = min(RAW_PULSE_MAX_US, max(RAW_PULSE_MIN_US, float(v[1])))
        duty = min(4095, max(0, ss._us2duty(us)))
        ss.pca9685.duty(chn, duty)
        continue

    if len(v) == 2 and v[1] == 'r':
        # release test mode: "<channel>,r\r" - cuts the pulse (0% duty) so
        # the servo goes limp. Used by pulse.py's Release button.
        chn = int(v[0])
        ss.release(chn)
        continue

    if len(v) != 2:
        print('syntax error', s)
        continue

    chn = int(v[0])
    deg = float(v[1])

    ss.position(chn, deg)

