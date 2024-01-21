import cv2 as p99_cv2
import mediapipe as p99_mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as p99_normalized_to_pixel_coords


def calculate_norm(point_a, point_b):
    """
    Calculate the norm of vector (two points: a and b).
    - (i1,j1) and (i2,j2): [(i1 - i2)^2 + (j1 - j2)^2]^0.5.
    Args:
        point_a: Point "a" of vector. [tuple[int, int]]
        point_b: Point "b" of vector. [tuple[int, int]]
    Returns:
        The norm of vector. [float]
    """
    return sum([(i - j) ** 2 for i, j in zip(point_a, point_b)]) ** 0.5


def calculate_ear(landmarks, chosen_idxs_landmarks, frame_w, frame_h):
    """
    Calculate the EYE ASPECT RATIO (EAR).
    Args:
        landmarks: Landmark detection list. [list]
        chosen_idxs_landmarks: The chosen positions of P1, P2, P3, P4, P5, P6. [list]
        frame_w: Width of frame (normalized). [int]
        frame_h: Height of frame (normalized). [int]
    Returns:
        The EAR. [float]
    """
    try:
        coord_points = []
        for chosen_idx in chosen_idxs_landmarks:
            current_landmark = landmarks[chosen_idx]
            normalized_coord = p99_normalized_to_pixel_coords(current_landmark.x, current_landmark.y, frame_w, frame_h)
            coord_points.append(normalized_coord)
        # CALCULATE THE NORM OF VECTORS
        p2_p6_norm = calculate_norm(coord_points[1], coord_points[5])
        p3_p5_norm = calculate_norm(coord_points[2], coord_points[4])
        p1_p4_norm = calculate_norm(coord_points[0], coord_points[3])
        # CALCULATE THE EAR
        ear = (p2_p6_norm + p3_p5_norm) / (2 * p1_p4_norm)
    except (Exception, ):
        ear = 0
        coord_points = None
    finally:
        return ear, coord_points


def calculate_ear_average(landmarks, left_chosen_idxs_ldmks, right_chosen_idxs_ldmks, frame_w, frame_h):
    """
    Calculate the EYE ASPECT RATIO (EAR) average.
    Args:
        landmarks: Landmark detection list. [list]
        left_chosen_idxs_ldmks: The chosen positions of P1, P2, P3, P4, P5, P6 (left eye). [list]
        right_chosen_idxs_ldmks: The chosen positions of P1, P2, P3, P4, P5, P6 (right eye). [list]
        frame_w: Width of frame (normalized). [int]
        frame_h: Height of frame (normalized). [int]
    Returns:
        The EAR average of two eyes. [float]
    """
    left_ear, left_coord_points = calculate_ear(landmarks, left_chosen_idxs_ldmks, frame_w, frame_h)
    right_ear, right_coord_points = calculate_ear(landmarks, right_chosen_idxs_ldmks, frame_w, frame_h)
    avg = (left_ear + right_ear) / 2
    return avg, (left_coord_points, right_coord_points)


def get_mediapipe_face_mesh():
    """
    Get the initialized MediaPipe face mesh object solution.
    Returns:
        The MediaPipe face mesh object. [FaceMesh]
    """
    return p99_mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=.5,
        min_tracking_confidence=.5,
    )


def draw_eye_landmarks(frame, left_landmark_coords, right_landmark_coords, color):
    """
    Draw eye landmarks in frame (image).
    Args:
        frame: The image to modify. [np.array]
        left_landmark_coords: The coords for the left eye. [list]
        right_landmark_coords: The coords for the right eye. [list]
        color: The color to plot. [BGR tuple. Example: (255, 255, 255)]
    Returns:
        The modified frame (with eye landmarks). [np.array]
    """
    for all_coords in [left_landmark_coords, right_landmark_coords]:
        if all_coords:
            for coord in all_coords:
                p99_cv2.circle(frame, coord, 2, color, -1)
    # FLIPPING THE FRAME HORIZONTALLY (SELFIE VIEW)
    frame = p99_cv2.flip(frame, 1)

    return frame


def draw_info_text(
    frame, text, text_position, color, font_type=p99_cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2
):
    """
    Draw text in frame (image).
    Args:
        frame: The image to modify. [np.array]
        text: Text to plot. [string]
        text_position: The coords to position the text. [list]
        color: The text color to plot. [BGR tuple. Example: (255, 255, 255)]
        font_type: Font type. [Enum cv2.HersheyFonts] [Default: cv2.FONT_HERSHEY_SIMPLEX].
        font_scale: The scale to display. [float] [Default: 0.8]
        thickness: Font thickness. [int] [Default: 2]
    Returns:
        The modified frame (with text). [np.array]
    """
    frame = p99_cv2.putText(frame, text, text_position, font_type, font_scale, color, thickness)
    return frame
