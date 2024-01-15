import os
import cv2
import numpy as np

class FaceDetector:
    def __init__(self, weights, dir='./', size=(0, 0)):
        # モデルを読み込む
        self.weights_path = os.path.join(dir, weights)
        self.face_detector = cv2.FaceDetectorYN.create(self.weights_path, "", size)

    def setInputSize(self, size: tuple):
        # 入力サイズを指定する
        self.face_detector.setInputSize(size)
    
    def detect(self, frame):
        # 顔を検出する
        _, faces = self.face_detector.detect(frame)
        faces = faces if faces is not None else []

        return faces
    
    def convert_ndaary2json(self, face: list):
        # データリストを辞書にする
        fase_data = list(map(int, face[:-1]))
        conf = float(face[-1])
        return {
            'bb': {
                'x': fase_data[0], 'y': fase_data[1], 'w': fase_data[2], 'h': fase_data[3]
            },
            'randmark': {
                'eye': {
                    'r_x': fase_data[4], 'r_y': fase_data[5], 'l_x': fase_data[6], 'l_y': fase_data[7]
                },
                'nose': {
                    'x': fase_data[8], 'y': fase_data[9]
                },
                'mouse': {
                    'r_x': fase_data[10], 'r_y': fase_data[11], 'l_x': fase_data[12], 'l_y': fase_data[13]
                }
            },
            'conf': conf
        }
    
    def convert_json2ndaary(self, face: dict):
        # 顔情報辞書をリストにする
        return np.array([
            *list(face['bb'].values()),
            *list(face['randmark']['eye'].values()),
            *list(face['randmark']['nose'].values()),
            *list(face['randmark']['mouse'].values()),
            face['conf']
        ])

    def draw_result(self, frame, faces: np.ndarray, bb=True, randmark=True, confidence=True):
        # 検出した顔のバウンディングボックスとランドマークを描画する
        for i, face in enumerate(faces):
            # 描画設定
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 0, 255)
            lineType = cv2.LINE_AA

            # バウンディングボックス
            if bb:
                box = list(map(int, face[:4]))
                cv2.rectangle(frame, box, color=color, thickness=2, lineType=lineType)

            # ランドマーク（右目、左目、鼻、右口角、左口角）
            if randmark:
                landmarks = list(map(int, face[4 : len(face)-1]))
                landmarks = np.array_split(landmarks, len(landmarks) / 2)
                for landmark in landmarks:
                    cv2.circle(frame, landmark, radius=2, color=color, thickness=-1, lineType=lineType)
                
            # 信頼度
            if confidence:
                conf = f"{round(float(face[-1]), 2)}"
                position = (int(face[0]), int(face[1]) - 10)
                cv2.putText(frame, conf, position, font, fontScale=0.5, color=color, thickness=2, lineType=lineType)
       