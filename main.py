import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import sys
from cvlib import Stream


def main():
    if '-camera' in sys.argv:
        name = 0
        input_type = Stream.INPUT_TYPE.CAMERA
    else:
        name = './test.jpg'
        input_type = Stream.INPUT_TYPE.FILE

    # ストリームインスタンスを作成する
    stream = Stream(name, input_type)

    # 顔検出器を登録
    stream.register_FD("onnx/yunet_n_320_320.onnx")

    # 顔認識器を登録
    stream.register_FR("onnx/face_recognizer_fast.onnx")

    # 特徴量を読み込む
    stream.load_feature_DB('./faces/feature')

    # 顔の画像と特徴量を保存する
    # stream.save_face('./faces')

    # 画像を描画
    stream.run(FD_flag=True, FR_flag=True, randmark=False)





if __name__ == '__main__':
    main()