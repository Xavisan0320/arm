import sys
import time
import math
import serial
import PySimpleGUI as sg
from util import nax, read_params, servo_angle_keys, radian, write_params, spin, get_move_time, SERVO_DEG_MIN, SERVO_DEG_MAX
import numpy as np


servo_angles = [np.nan] * nax

# One flag per channel so a command that runs past the end of the servo's
# travel is reported once, not on every step of the move.
servo_over_travel = [False] * nax

def angle_limits(ch):
    """The arm-degrees of channel ch that map inside the servo's travel."""
    lo = servo_to_angle(ch, SERVO_DEG_MIN)
    hi = servo_to_angle(ch, SERVO_DEG_MAX)

    return (min(lo, hi), max(lo, hi))

def init_servo_nano(params):
    global pwm
    try:
        import Adafruit_PCA9685

        pwm = Adafruit_PCA9685.PCA9685(address=0x40)
        pwm.set_pwm_freq(60)

    except ModuleNotFoundError:
        print("no Adafruit")

def setPWM(ch, pos):
    pulse = round( 150 + (600 - 150) * (pos + 0.5 * math.pi) / math.pi )
    pwm.set_pwm(ch, 0, pulse)

def set_servo_angle_nano(ch, deg):
    rad = radian(deg)
    setPWM(ch, rad)


def init_servo(params_arg):
    global params, servo_angles, servo_param, ser

    params = params_arg

    for ch, deg in enumerate(params['prev-servo']):
        servo_angles[ch] = deg

    com_port = params['COM'] 
    servo_param = params['servo-angle']

    try:
        ser = serial.Serial(com_port, 115200, timeout=1, write_timeout=1)
    except serial.serialutil.SerialException: 
        print(f'The specified serial port does not exist.{com_port}')
        sys.exit(0)

def set_servo_param(ch, scale, offset):
    servo_param[ch] = [scale, offset]
    params['servo-angle'][ch] = [scale, offset]

    write_params(params)

def angle_to_servo(ch, deg):
    coef, intercept = servo_param[ch]

    return coef * deg + intercept

def servo_to_angle(ch, deg):
    coef, intercept = servo_param[ch]

    return (deg - intercept) / coef

def set_servo_angle(ch : int, deg : float):
    if deg < SERVO_DEG_MIN or SERVO_DEG_MAX < deg:
        if not servo_over_travel[ch]:
            servo_over_travel[ch] = True
            lo, hi = angle_limits(ch)
            print(f'J{ch+1}: {deg:.1f} is past the servo travel '
                  f'{SERVO_DEG_MIN}..{SERVO_DEG_MAX} (arm {lo:.0f}..{hi:.0f} deg) - '
                  f'the firmware will clamp it and the arm will not hold the pose asked for')
    else:
        servo_over_travel[ch] = False

    servo_angles[ch] = deg

    cmd = "%d,%.1f\r" % (ch, deg)

    while True:
        try:
            n = ser.write(cmd.encode('utf-8'))
            break
        except serial.SerialTimeoutException:
            print("write time out")
            time.sleep(1)

    ret = ser.read_all().decode('utf-8')
    if "error" in ret:
        print("read", ret)

def move_servo(ch, dst):
    src = servo_angles[ch]

    move_time = get_move_time()
    start_time = time.time()
    while True:
        t = (time.time() - start_time) / move_time
        if 1 <= t:
            break

        deg = t * dst + (1 - t) * src

        set_servo_angle(ch, deg)

        yield

def move_all_servo(dsts):
    srcs = list(servo_angles)

    move_time = get_move_time()
    start_time = time.time()

    while True:
        t = (time.time() - start_time) / move_time
        if 1 <= t:
            break

        for ch in range(nax):

            deg = t * dsts[ch] + (1 - t) * srcs[ch]

            set_servo_angle(ch, deg)

        yield

if __name__ == '__main__':

    params = read_params()

    init_servo(params)

    layout = [
        [
            spin(f'J{i+1}', f'J{i+1}-servo', int(servo_angles[i]), SERVO_DEG_MIN, SERVO_DEG_MAX, True) for i in range(nax)
        ]
        +
        [ sg.Text('', size=(15,1)) ]
        +
        [ sg.Button('Close') ]
    ]

    window = sg.Window('Servo', layout, element_justification='c') # disable_minimize=True, 

    moving = None

    while True:
        event, values = window.read(timeout=1)

        if moving is not None:
            try:
                moving.__next__()

            except StopIteration:
                moving = None
                print('========== stop moving ==========')

                params['prev-servo'] = servo_angles
                write_params(params)

        if event in servo_angle_keys:
            ch = servo_angle_keys.index(event)

            # A Spin hands back the raw string when what was typed is not one of
            # its listed choices, so anything non-numeric has to be dropped here
            # rather than fed into the interpolation as a string.
            try:
                deg = float(values[event])
            except ValueError:
                continue

            moving = move_servo(ch, deg)

        elif event == sg.WIN_CLOSED or event == 'Close':

            params['prev-servo'] = servo_angles
            write_params(params)

            break

    window.close()

