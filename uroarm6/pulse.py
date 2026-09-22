"""
pulse.py - raw servo pulse-width tester.

Standalone bring-up tool: sends a raw pulse width straight to any PCA9685
channel (0-15) on the servo driver, bypassing the arm-degree calibration in
data/arm.json entirely. Use this to check a newly-mounted servo's
travel/centre before running angle.py to (re-)derive its servo-angle linear
fit.

500-2500us has been confirmed safe (no stalling) for the DS3218 servos this
tool was built for - the slider is kept to that range. The direct-input
boxes additionally accept up to 2600us, to probe for extra travel at the
upper end only - watch/listen for the servo stalling or buzzing and back
off immediately if it does; don't hold it against a mechanical stop.

Protocol (see pico/main.py):
    "<channel>,<microseconds>,u\r"  -> drive channel to that pulse width
    "<channel>,r\r"                 -> release (cut pulse, servo goes limp)

This talks to the Pico directly over serial - it does not import servo.py's
init_servo()/set_servo_angle(), since those work in arm-degrees and would
read/write the live data/arm.json calibration, which is not meaningful yet
for an uncalibrated/replacement servo.
"""

import sys
import time
import serial
import PySimpleGUI as sg
from util import read_params

NUM_CH = 16

# Confirmed-safe range (matches pico/main.py's Servos min_us/max_us for the
# normal degree path). The slider is kept to this range only.
SAFE_MIN_US = 500
SAFE_MAX_US = 2500

# The direct-input boxes allow a bit more headroom on the upper end only,
# to probe for extra travel there. Must stay within pico/main.py's
# RAW_PULSE_MIN_US/MAX_US clamp or the firmware will silently clamp beyond
# what's typed here.
MIN_US = SAFE_MIN_US
MAX_US = 2600

CENTER_US = 1500

ENTER_SUFFIX = '_Enter'


def open_serial(com_port):
    try:
        return serial.Serial(com_port, 115200, timeout=1, write_timeout=1)
    except serial.serialutil.SerialException:
        print(f'The specified serial port does not exist. {com_port}')
        sys.exit(0)


def send_pulse(ser, ch, us):
    us = min(MAX_US, max(MIN_US, us))
    cmd = "%d,%.0f,u\r" % (ch, us)

    while True:
        try:
            ser.write(cmd.encode('utf-8'))
            break
        except serial.SerialTimeoutException:
            print("write time out")
            time.sleep(1)

    ret = ser.read_all()
    if ret and "error" in ret.decode('utf-8', errors='ignore'):
        print("read", ret)

    return us


def send_release(ser, ch):
    cmd = "%d,r\r" % ch

    while True:
        try:
            ser.write(cmd.encode('utf-8'))
            break
        except serial.SerialTimeoutException:
            print("write time out")
            time.sleep(1)

    ser.read_all()


def slider_key(ch):
    return f'-pulse-{ch}-'


def input_key(ch):
    return f'-input-{ch}-'


def release_key(ch):
    return f'-release-{ch}-'


def center_key(ch):
    return f'-center-{ch}-'


def set_display(window, ch, us):
    """Update both the slider and the input box for ch to us, without
    re-triggering their change events. The slider only spans
    SAFE_MIN_US-SAFE_MAX_US, so a value entered above that (via the direct
    input) is shown clamped on the slider but exact in the input box."""
    slider_us = min(SAFE_MAX_US, max(SAFE_MIN_US, us))
    window[slider_key(ch)].update(slider_us)
    window[input_key(ch)].update(int(us))


def channel_row(ch):
    return [
        sg.Text(f'CH{ch:<2}', size=(4, 1)),
        sg.Slider(
            range=(SAFE_MIN_US, SAFE_MAX_US),
            default_value=CENTER_US,
            resolution=1,
            orientation='h',
            size=(36, 15),
            key=slider_key(ch),
            enable_events=True,
        ),
        sg.Input(
            str(CENTER_US),
            key=input_key(ch),
            size=(6, 1),
            justification='r',
        ),
        sg.Text('us'),
        sg.Button('Center', key=center_key(ch), size=(7, 1)),
        sg.Button('Release', key=release_key(ch), size=(7, 1)),
    ]


if __name__ == '__main__':
    params = read_params()
    com_port = params['COM']

    ser = open_serial(com_port)

    layout = [
        [sg.Text(
            f'Pulse tester - {com_port}   '
            f'slider (safe): {SAFE_MIN_US}-{SAFE_MAX_US}us   '
            f'direct input: {MIN_US}-{MAX_US}us (DS3218 180deg)'
        )],
        [sg.Text(
            f'Direct input above {SAFE_MAX_US}us is beyond the confirmed-safe '
            'range: watch/listen for stalling or buzzing and back off immediately.',
            text_color='orange'
        )],
    ]
    layout += [channel_row(ch) for ch in range(NUM_CH)]
    layout += [
        [
            sg.Button('All Center'),
            sg.Button('Release All'),
            sg.Push(),
            sg.Button('Close'),
        ]
    ]

    window = sg.Window('Pulse Tester', layout, element_justification='c', finalize=True)

    # Enter key in an input box applies that channel's value immediately,
    # without needing a separate per-row "Set" button.
    for ch in range(NUM_CH):
        window[input_key(ch)].bind('<Return>', ENTER_SUFFIX)

    while True:
        event, values = window.read(timeout=50)

        if event == sg.WIN_CLOSED or event == 'Close':
            break

        elif event == 'All Center':
            for ch in range(NUM_CH):
                set_display(window, ch, CENTER_US)
                send_pulse(ser, ch, CENTER_US)

        elif event == 'Release All':
            for ch in range(NUM_CH):
                send_release(ser, ch)

        elif event.startswith('-center-'):
            ch = int(event.split('-')[2])
            set_display(window, ch, CENTER_US)
            send_pulse(ser, ch, CENTER_US)

        elif event.startswith('-release-'):
            ch = int(event.split('-')[2])
            send_release(ser, ch)

        elif event.startswith('-pulse-'):
            ch = int(event.split('-')[2])
            us = send_pulse(ser, ch, values[event])
            window[input_key(ch)].update(int(us))

        elif event.endswith(ENTER_SUFFIX) and event[:-len(ENTER_SUFFIX)].startswith('-input-'):
            base_key = event[:-len(ENTER_SUFFIX)]
            ch = int(base_key.split('-')[2])
            try:
                us = float(values[base_key])
            except ValueError:
                continue
            us = send_pulse(ser, ch, us)
            set_display(window, ch, us)

    window.close()
    ser.close()
