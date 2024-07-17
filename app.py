import argparse
import yaml
from coordernadas_vaga import Coordenadas
from detectar_carro import Detector
from RGB import *
import logging


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    image_file = args.image_file
    data_file = args.data_file
    start_frame = args.start_frame

    if image_file is not None:
        with open(data_file, "w+") as points:
            generator = Coordenadas(image_file, points, RED)
            generator.generate()

    with open(data_file, "r") as data:
        points = yaml.load(data, Loader=yaml.FullLoader)
        detector = Detector(args.video_file, points, int(start_frame))
        detector.detect_motion()


def parse_args():
    parser = argparse.ArgumentParser(description='Generates Coordinates File')

    parser.add_argument("--image",
                        dest="image_file",
                        required=False,
                        help="Image file to generate coordinates on")

    parser.add_argument("--video",
                        dest="video_file",
                        required=True,
                        help="Video file to detect motion on")

    parser.add_argument("--data",
                        dest="data_file",
                        required=True,
                        help="Data file to be used with OpenCV")

    parser.add_argument("--start-frame",
                        dest="start_frame",
                        required=False,
                        default=1,
                        help="Starting frame on the video")

    return parser.parse_args()


if __name__ == '__main__':
    main()
