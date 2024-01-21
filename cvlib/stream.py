import sys
import cv2
import numpy as np
import glob
from pathlib import Path

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

        # 特徴量リストを初期化する
        self.reset_feature_DB()
    
    def load_feature_DB(self, path='./'):
        # パスの存在確認.無ければ戻る.
        p_path = Path(path)
        if not p_path.exists():
            return

        # pathのリストを用意
        if p_path.is_file():
            files = [str(p_path)]
        else:
            files = glob.glob(p_path.joinpath('*.npy'))
        
        # npyファイルを読み込む
        for file in files:
            self.feature_DB.append({
                'userID': Path(file).stem,
                'feature': np.load(file)
            })
    
    def reset_feature_DB(self):
        # 特徴量リストを初期化する
        self.feature_DB = []
    
    def remove_feature_DB(self, key: int | str):
        # DBが存在しなければ戻る
        if not self.feature_DB:
            return

        # key指定がインデックスの場合
        if type(key) is int and key < len(self.feature_DB):
            self.feature_DB.pop(key)
        
        # key指定が文字列（userID）の時
        elif type(key) is str:
            for i, feature in enumerate(self.feature_DB):
                if feature['userID'] == key:
                    self.feature_DB.pop(i)
                    break

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
    
    def recognize(self):
        if self.face_recognizer:
            # 顔を検出する
            faces = self.detect()

            # 検出された顔を切り抜く
            aligned_faces = self.face_recognizer.faces_alignCrop(self.frame, faces)
            
            # 特徴を抽出する
            features = self.face_recognizer.get_features(aligned_faces)
            
            # DBと照合する
            return self.face_recognizer.recognize(features, self.feature_DB)
        
        else:
            return []
    
    def save_face(self, dir: str, image_dir='img', feature_dir='feature',
                  image_name_list: list = None, feature_name_list: list = None):
        # image_dir, feature_dirどちらか指定されている場合
        if self.face_recognizer and (image_dir or feature_dir):
            # dirのパスのディレクトリが存在することを確認し、無ければ作成する
            p_file = Path(dir)
            if not p_file.exists():
                p_file.mkdir()

            # 顔を検出する
            faces = self.detect()

            # 顔画像を切り抜く
            aligned_faces = self.face_recognizer.faces_alignCrop(self.frame, faces)

            # 顔画像を保存する
            if image_dir:
                self.face_recognizer.save_faces(p_file.joinpath(image_dir), aligned_faces, image_name_list)

            # 特徴量を保存する
            if feature_dir:
                features = self.face_recognizer.get_features(aligned_faces)
                self.face_recognizer.save_features(p_file.joinpath(feature_dir), features, feature_name_list)

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
