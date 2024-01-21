import time as p99_time
import cv2 as p99_cv2
import drowsiness_utils as p99_drowsiness_utils


class DrowsinessModel:
    def __init__(self):
        self.landmarks_idxs = {
            'left_eye': [362, 385, 387, 263, 373, 380],
            'right_eye': [33, 160, 158, 133, 153, 144],
        }
        self.mp_face_mesh_model = p99_drowsiness_utils.get_mediapipe_face_mesh()
        self.detection_state = {
            'start_time': p99_time.perf_counter(),
            'drowsy_time': 0,
            'emit_alarm': False,
        }

    def _reset_detection_state(self):
        self.detection_state['start_time'] = p99_time.perf_counter()
        self.detection_state['drowsy_time'] = 0
        self.detection_state['emit_alarm'] = False

    def make_detections(self, frame, wait_time, ear_threshold, depure=False):
        frame.flags.writeable = False  # ONLY READONLY, INMUTABLE FRAME TO PASS BY REFERENCE
        img_height, img_width, channels = frame.shape
        detections = self.mp_face_mesh_model.process(frame)
        # ALGORITHM FOR FACE DETECTION
        if detections.multi_face_landmarks:
            landmarks = detections.multi_face_landmarks[0].landmark
            ear, coords = p99_drowsiness_utils.calculate_ear_average(
                landmarks,
                self.landmarks_idxs['left_eye'],
                self.landmarks_idxs['right_eye'],
                img_width,
                img_height,
            )

            if depure:
                # ONLY IF depure=True, PRINT THE EYE LANDMARKS
                frame = p99_drowsiness_utils.draw_eye_landmarks(frame, coords[0], coords[1], (0, 255, 0))

            if ear < ear_threshold:
                # INCREASE drowsy_time TO TRACK THE TIME WITH EYES CLOSED AND RESET start_time
                end_time = p99_time.perf_counter()
                self.detection_state['drowsy_time'] += (end_time - self.detection_state['start_time'])
                self.detection_state['start_time'] = end_time
                if self.detection_state['drowsy_time'] >= wait_time:
                    # STATE: THE PERSON IS ASLEEP
                    self.detection_state['emit_alarm'] = True
                    if depure:
                        # ONLY IF depure=True, DRAW INFO TEXT
                        p99_drowsiness_utils.draw_info_text(
                            frame,
                            'WARNING: DROWSINESS',
                            (16, int(img_height // 2 * 1.7)),
                            (0, 0, 255),
                        )
            else:
                self._reset_detection_state()

            if depure:
                # ONLY IF depure=True, DRAW INFO TEXT
                p99_drowsiness_utils.draw_info_text(
                    frame,
                    f'EAR -> {ear}',
                    (16, 32),
                    (0, 255, 0),
                )
                p99_drowsiness_utils.draw_info_text(
                    frame,
                    f'Drowsy: {round(self.detection_state["drowsy_time"]), wait_time} seconds',
                    (16, int(img_height // 2 * 1.5)),
                    (0, 255, 0),
                )
        else:
            self._reset_detection_state()
            # FLIPPING THE FRAME HORIZONTALLY (SELFIE VIEW)
            frame = p99_cv2.flip(frame, 1)

        return frame, self.detection_state['emit_alarm']
