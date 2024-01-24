import os
import cv2
import numpy as np
from pathlib import Path


class FaceRecognizer:
    COSINE_THRESHOLD = 0.363
    NORML2_THRESHOLD = 1.128

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
    
    def save_faces(self, dir: str, aligned_faces: list, name_list: list = None):
        # dirのパスのディレクトリが存在することを確認し、無ければ作成する
        p_file = Path(dir)
        if not p_file.exists():
            p_file.mkdir()

        # 指定がない場合、保存名を生成
        if name_list is None:
            name_list = [f'face{(i + 1):03}.jpg' for i in range(len(aligned_faces))]

        # 顔画像を保存する
        for name, aligned_face in zip(name_list, aligned_faces):
            cv2.imwrite(os.path.join(dir, name), aligned_face)
    
    def save_features(self, dir: str, features: list, name_list: list = None):
        # dirのパスのディレクトリが存在することを確認し、無ければ作成する
        p_file = Path(dir)
        if not p_file.exists():
            p_file.mkdir()

        # 指定がない場合、保存名を生成
        if name_list is None:
            name_list = [f'feature{(i + 1):03}' for i in range(len(features))]

        # 特徴を保存する　
        for name, feature in zip(name_list, features):
            np.save(os.path.join(dir, name), feature)
    
    def recognize(self, features: list, feature_DB: list, must_result=False):
        return [self.match(feature, feature_DB, must_result) for feature in features]

    def match(self, feature, feature_DB, must_result=False):
        # 特徴を辞書と比較してマッチしたユーザーとスコアを返す関数
        best_match = {'userID': '', 'score': 0.0}
        for info in feature_DB:
            score = self.face_recognizer.match(feature, info['feature'], cv2.FaceRecognizerSF_FR_COSINE)
            if best_match['score'] < score:
                best_match['userID'] = info['userID']
                best_match['score'] = score

        if must_result:
            return best_match
        
        else:
            if best_match['score'] > self.COSINE_THRESHOLD:
                return best_match
            else:
                return {'userID': '', 'score': 0.0}

    def draw_result(self, frame, faces: np.ndarray, recognize_results: list, bb=True, recognition=True):
        # 描画設定
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 255)
        lineType = cv2.LINE_AA

        for face, recognize_result in zip(faces, recognize_results):
            # バウンディングボックス
            if bb:
                box = list(map(int, face[:4]))
                cv2.rectangle(frame, box, color=color, thickness=2, lineType=lineType)

            # 認識の結果を描画する
            if recognition:
                userID = recognize_result['userID'] if recognize_result['userID'] else 'unknown'
                score = recognize_result['score']
                text = f"{userID} ({round(score, 3)})"
                box = list(map(int, face[:2]))
                position = (box[0], box[1] - 10)
                cv2.putText(frame, text, position, font, fontScale=0.6, color=color, thickness=2, lineType=lineType)
    
        return frame
