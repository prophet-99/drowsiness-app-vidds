import threading as p99_threading
import time as p99_time
from datetime import datetime as p99_datetime

import cv2 as p99_cv2
import requests as p99_requests
from playsound import playsound as p99_playsound

from drowsiness_model import DrowsinessModel
from GPS_manager import GPSManager


# GET CURRENT INSTANT + DATE IN ISO 8601
def get_current_instant():
    current_date = p99_datetime.now()
    return current_date.strftime('%Y-%m-%dT%H:%M:%S.%f')


# MAKE REQUEST TO API
def make_request(url, payload):
    response = p99_requests.post(url, json=payload)
    print(response.json())


if __name__ == '__main__':
    # GPS MANAGER - LOCAL VARIABLES
    gps_manager_instance = GPSManager()
    # DROWSINESS MODEL - LOCAL VARIABLES
    cap = p99_cv2.VideoCapture(0)
    drowsiness_model_instance = DrowsinessModel()
    detection_state = {
        'start_time': p99_time.perf_counter(),
        'alarm_time': 0,
        'alarm_threshold': 3,
        'play_sound': False,
    }

    while True:
        _, image = cap.read()
        # CONFIGURE DROWSINESS MODEL (THRESHOLD FOR EAR: 0.18)
        image, emit_alarm = drowsiness_model_instance.make_detections(
            image,
            wait_time=1,
            ear_threshold=0.18,
            depure=True,
        )
        p99_cv2.imshow('Drowsiness detection test', image)
        # SOUND MODULE
        if emit_alarm:
            sc_perf_counter = p99_time.perf_counter()
            detection_state['alarm_time'] += sc_perf_counter - detection_state['start_time']
            detection_state['start_time'] = sc_perf_counter
            if not detection_state['play_sound']:
                try:
                    p99_playsound('sounds/alarm.mp3', block=False)
                    # REQUEST TO API (drowsyTime, gpsCoords)
                    thread = p99_threading.Thread(
                        target=make_request,
                        args=(
                            'http://localhost:3000/api/v1/statistics',
                            {
                                'drowsyInstant': get_current_instant(),
                                'gpsCoords': gps_manager_instance.get_current_coords()
                            },
                        )
                    )
                    thread.start()
                except Exception as e:
                    print(e.args[0])
                    print('Error with playsound')
                print('DROWSINESS')
                detection_state['play_sound'] = True
            if detection_state['alarm_time'] >= detection_state['alarm_threshold']:
                detection_state['play_sound'] = False
                detection_state['alarm_time'] = 0
                detection_state['start_time'] = p99_time.perf_counter()
        else:
            detection_state['play_sound'] = False
            detection_state['alarm_time'] = 0
            detection_state['start_time'] = p99_time.perf_counter()

        if p99_cv2.waitKey(1) == ord('q'):
            break

    # RELEASE CV2
    cap.release()
    p99_cv2.destroyAllWindows()
    # RELEASE GPS_manager
    gps_manager_instance.stop_updating_coords()
