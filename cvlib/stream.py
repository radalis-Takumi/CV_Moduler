import sys
import cv2
import numpy as np

from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer


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

            # 初期フレームを設定
            self.frame = np.full((self.height, self.width, self.channel), 255, dtype=np.uint8)

    def register_FD(self, weights: str, dir='./'):
        # モデルを読み込む
        self.face_detector = FaceDetector(weights, dir, (self.width, self.height))
    
    def register_FR(self, weights: str, dir='./'):
        # モデルを読み込む
        self.face_recognizer = FaceRecognizer(weights, dir)

    def detect(self, json=False):
        if self.face_detector:
            # 顔を検出する
            faces = self.face_detector.detect(self.frame)

            # JSONにする
            if json:
                faces = [self.face_detector.convert_ndaary2json(face) for face in faces]
                
            return faces
    
        else:
            return []
    
    def save_face(self, dir: str, image_dir='img', feature_dir='feature', name_list: list = None):
        if self.face_recognizer:
            # 顔を検出する
            faces = self.detect()

            # 顔画像を切り抜いて保存する
            aligned_faces = self.face_recognizer.faces_alignCrop(self.frame, faces)
            self.face_recognizer.save_face(dir, aligned_faces, name_list)


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
