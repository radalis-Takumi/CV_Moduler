import os
import cv2
import numpy as np

class FaceRecognizer:
    def __init__(self, weights, dir='./'):
        # モデルを読み込む
        self.weights_path = os.path.join(dir, weights)
        self.face_recognizer = cv2.FaceRecognizerSF.create(self.weights_path, "")
    
    def faces_alignCrop(self, frame, faces: list):
        # 検出された顔を切り抜く
        return [self.face_recognizer.alignCrop(frame, face) for face in faces]
    
    def get_features(self, aligned_faces: list):
        # 特徴を抽出する
        features = [self.face_recognizer.feature(aligned_face) for aligned_face in aligned_faces]

        return features
    
    def save_face(self, dir: str, aligned_faces: list, name_list: list = None):
        # 指定がない場合、保存名を生成
        if name_list is None:
            name_list = [f'face{(i + 1):03}.jpg' for i in range(len(aligned_faces))]

        # 顔画像を保存する
        for name, aligned_face in zip(name_list, aligned_faces):
            cv2.imwrite(os.path.join(dir, name), aligned_face)
    
    def save_features(self, dir: str, features: list, name_list: list = None):
        # 指定がない場合、保存名を生成
        if name_list is None:
            name_list = [f'feature{(i + 1):03}' for i in range(len(features))]

        # 特徴を保存する　
        for name, feature in zip(name_list, features):
            np.save(os.path.join(dir, name), feature)
