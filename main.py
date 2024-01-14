import sys
from cvlib import Stream


def main():
    if '-camera' in sys.argv:
        name = 0
        input_type = Stream.INPUT_TYPE.CAMERA
    else:
        name = './test.jpg'
        input_type = Stream.INPUT_TYPE.FILE

    stream = Stream(name, input_type)


if __name__ == '__main__':
    main()