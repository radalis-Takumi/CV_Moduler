import os
import sys
import cv2
import numpy as np


class Stream:
    class INPUT_TYPE:
        FILE = 0
        CAMERA = 1
        RSTP = 2

    def __init__(self, name: str | int, input_type=INPUT_TYPE.FILE):
        self.input_type = input_type

        #  画像を入力するとき
        if input_type == self.INPUT_TYPE.FILE:
            # 画像の読み込み
            self.frame = cv2.imread(name)

            # 画像が開かなかったとき終了する
            if self.frame is None:
                sys.exit()

            # 入力サイズを取得する
            self.height, self.width, self.channel = self.frame.shape

        # webカメラを入力するとき
        elif input_type == self.INPUT_TYPE.CAMERA:
            self.capture = cv2.VideoCapture(name)

            # カメラが開かなかったとき終了する
            if not self.capture.isOpened():
                sys.exit()
            
            # 入力サイズを取得する
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.channel = 3

    def register_FD(self, weights, dir='./'):
        # モデルを読み込む
        self.face_detector = FaceDetector(weights, dir, (self.width, self.height))

    def run(self, window_name='window', windowSizeVariable=False, FD_flag=False, delay=1):
        # フレームの描画を行う
        self.__preset_run(window_name, windowSizeVariable, FD_flag)

        if self.input_type == self.INPUT_TYPE.FILE:
            self.__run_image()
        
        elif self.input_type == self.INPUT_TYPE.CAMERA:
            self.__run_video(delay)
    
    def __preset_run(self, window_name, windowSizeVariable, FD_flag):
        self.window_name = window_name
        self.FD_flag = FD_flag
        windowFlag = cv2.WINDOW_NORMAL if windowSizeVariable else cv2.WINDOW_AUTOSIZE
        cv2.namedWindow(self.window_name, windowFlag)
    
    def __isWindowExist(self):
        try:
            cv2.getWindowProperty(self.window_name, cv2.WND_PROP_AUTOSIZE)
            return True
        except:
            return False

    def __run_image(self):
        if self.FD_flag and self.face_detector:
            # 顔を検出する
            faces = self.face_detector.detect(self.frame)

            # 検出した顔のバウンディングボックスとランドマークを描画する
            self.face_detector.draw_result(self.frame, faces)

        cv2.imshow(self.window_name, self.frame)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
    
    def __run_video(self, delay=1):
        while True:
            # フレームをキャプチャして画像を読み込む
            res, self.frame = self.capture.read()
            if res:
                if self.FD_flag and self.face_detector:
                    # 顔を検出する
                    faces = self.face_detector.detect(self.frame)

                    # 検出した顔のバウンディングボックスとランドマークを描画する
                    self.face_detector.draw_result(self.frame, faces)

                cv2.imshow(self.window_name, self.frame)
                cv2.waitKey(delay)
                if not self.__isWindowExist():
                    break
            else:
                break

        cv2.destroyAllWindows()


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
        
