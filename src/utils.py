```python
import cv2
import numpy as np
import math
import mediapipe as mp
import os

LANDMARK_COLOR = (0, 255, 0)
CONNECTION_COLOR = (0, 0, 255)
THICKNESS = 2
CIRCLE_RADIUS = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)
UI_BG_COLOR = (50, 50, 50)
UI_TEXT_POS = (10, 30)
MODEL_NAME_POS = (10, 60)

FIST_THRESHOLD_WRIST_MCP = 0.35
PINCH_THRESHOLD = 0.06

TRANSLATION_SENSITIVITY = 1.5
ROTATION_SENSITIVITY = 1.0
SCALE_SENSITIVITY = 1.0

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def calculate_distance(p1, p2):
    if p1 is None or p2 is None:
        return float('inf')
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_distance_2d(p1, p2, image_shape):
    if p1 is None or p2 is None:
        return float('inf')
    h, w = image_shape[:2]
    p1_px = (int(p1.x * w), int(p1.y * h))
    p2_px = (int(p2.x * w), int(p2.y * h))
    return math.sqrt((p1_px[0] - p2_px[0])**2 + (p1_px[1] - p2_px[1])**2)

def normalize_coordinates(landmark, image_shape):
    if landmark is None:
        return None, None
    h, w = image_shape[:2]
    cx, cy = int(landmark.x * w), int(landmark.y * h)
    return cx, cy

def map_to_range(value, in_min, in_max, out_min, out_max):
    if in_max == in_min: return out_min
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def vector_subtraction(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

def vector_magnitude(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def normalize_vector(v):
    mag = vector_magnitude(v)
    if mag == 0: return (0, 0, 0)
    return (v[0] / mag, v[1] / mag, v[2] / mag)

def get_hand_landmarks(results, hand_index=0):
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > hand_index:
        return results.multi_hand_landmarks[hand_index]
    return None

def get_handedness(results, hand_index=0):
    if results.multi_handedness and len(results.multi_handedness) > hand_index:
        return results.multi_handedness[hand_index].classification[0].label
    return None

def get_hand_center(hand_landmarks, image_shape):
    if not hand_landmarks:
        return None
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    mcp_middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    h, w = image_shape[:2]
    center_x = int(((wrist.x + mcp_middle.x) / 2) * w)
    center_y = int(((wrist.y + mcp_middle.y) / 2) * h)
    center_z = wrist.z
    return center_x, center_y, center_z

def is_fist(hand_landmarks):
    if not hand_landmarks:
        return False

    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    finger_tips_indices = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_mcp_indices = [
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]

    finger_tips = [hand_landmarks.landmark[i] for i in finger_tips_indices]
    finger_mcps = [hand_landmarks.landmark[i] for i in finger_mcp_indices]

    avg_dist_to_wrist = sum(calculate_distance(tip, wrist) for tip in finger_tips) / len(finger_tips)

    mcp_middle = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ref_dist = calculate_distance(wrist, mcp_middle)

    if ref_dist > 0 and (avg_dist_to_wrist / ref_dist) < FIST_THRESHOLD_WRIST_MCP:
        mcp_y_avg = sum(mcp.y for mcp in finger_mcps) / len(finger_mcps)
        tip_y_avg = sum(tip.y for tip in finger_tips[1:]) / (len(finger_tips) - 1) # Exclude thumb

        # Check if fingertips are generally lower (higher y) than MCPs
        # And also check if fingertips are close to each other (optional, adds robustness)
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        tip_spread = calculate_distance(index_tip, pinky_tip)

        # Adjust threshold based on hand size reference (e.g., wrist to middle MCP distance)
        spread_threshold = ref_dist * 0.8 # Example threshold for tip spread

        if tip_y_avg > mcp_y_avg and tip_spread < spread_threshold:
             return True

    return False

def get_pinch_distance(hand_landmarks):
    if not hand_landmarks:
        return float('inf')

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    return calculate_distance(thumb_tip, index_tip)

def get_two_hand_scale_factor(results, prev_distance):
    if not results.multi_hand_landmarks or len(results.multi_hand_landmarks) < 2:
        return 1.0, prev_distance

    hand1_landmarks = results.multi_hand_landmarks[0]
    hand2_landmarks = results.multi_hand_landmarks[1]

    wrist1 = hand1_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    wrist2 = hand2_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    current_distance = calculate_distance(wrist1, wrist2)

    if prev_distance is None or prev_distance <= 0:
        return 1.0, current_distance

    # Use relative change to avoid extreme jumps with noisy data
    scale_factor = 1.0 + (current_distance - prev_distance) * SCALE_SENSITIVITY

    # Clamp scale factor for stability
    scale_factor = max(0.85, min(1.15, scale_factor)) # Gentle scaling per frame

    # Alternative: Direct ratio (can be jumpy)
    # scale_factor = current_distance / prev_distance
    # scale_factor = max(0.5, min(2.0, scale_factor))

    return scale_factor, current_distance


def draw_landmarks(image, hand_landmarks):
    if hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

def draw_all_landmarks(image, results):
     if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_landmarks(image, hand_landmarks)

def draw_ui(image, mode, model_name):
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), UI_BG_COLOR, -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    cv2.putText(image, f"Mode: {mode}", UI_TEXT_POS, FONT, FONT_SCALE, FONT_COLOR, THICKNESS, cv2.LINE_AA)
