import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def transform_video(video, name:str):
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2) as holistic:
        _process_video(video, holistic)

def _process_video(video, holistic):
    while True:
        ret, frame = video.read()
        if ret == False:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        _draw_landmarks(frame, results,"RHand")
        _draw_landmarks(frame, results,"LHand")
        _draw_landmarks(frame, results,"Pose")
        _draw_landmarks(frame, results,"Face")
    
    video.release()
    cv2.destroyAllWindows()


def _return_configurations(type:str,results):
    configuration_draw = {
        "RHand":{
            "results":results.right_hand_landmarks,
            "connections": mp_holistic.HAND_CONNECTIONS,
            "drawing1": mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            "drawing2":mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        },
        "LHand":{
            "results":results.left_hand_landmarks,
            "connections": mp_holistic.HAND_CONNECTIONS,
            "drawing1":  mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            "drawing2":mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        },
        "Pose":{
            "results":results.left_hand_landmarks,
            "connections": mp_holistic.POSE_CONNECTIONS,
            "drawing1": mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            "drawing2":mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        },
        "Face":{
            "results":results.face_landmarks,
            "connections": mp_holistic.FACEMESH_TESSELATION,
            "drawing1": mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            "drawing2":mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        }
    }
    
    return configuration_draw[type]

def _draw_landmarks(frame, results, type:str):
    configuration = _return_configurations(type,results)
    mp_drawing.draw_landmarks(frame, configuration["results"], configuration["connections"], 
                                    configuration["drawing1"], 
                                    configuration["drawing2"]
                                    )


    
