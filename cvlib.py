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
        #  画像を入力するとき
        if input_type == self.INPUT_TYPE.FILE:
            # 画像の読み込み
            self.frame = cv2.imread(name)

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
        self.face_detector = FaceDetector(weights, dir)

        # 入力サイズを指定する
        self.face_detector.setInputSize((self.width, self.height))


class FaceDetector:
    def __init__(self, weights, dir='./'):
        # モデルを読み込む
        self.weights_path = os.path.join(dir, weights)
        self.face_detector = cv2.FaceDetectorYN_create(self.weights_path, "", (0, 0))

    def setInputSize(self, size: tuple):
        # 入力サイズを指定する
        self.face_detector.setInputSize(size)
