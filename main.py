import argparse

import cv2
import numpy as np
from utils.generate import generateTextureMap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--image_path",
        required=True,
        type=str,
        help="path of image you want to quilt",
    )
    parser.add_argument(
        "-b", "--block_size", type=int, help="block size in pixels", required=True
    )
    parser.add_argument(
        "-o", "--overlap", type=int, help="overlap size in pixels", required=True
    )
    parser.add_argument(
        "-s", "--scale", type=float, default=2, help="Scaling w.r.t. to image size"
    )
    parser.add_argument(
        "-n",
        "--num_outputs",
        type=int,
        default=1,
        help="number of output textures required",
    )
    parser.add_argument(
        "-f", "--output_file", type=str, help="output file name", required=True
    )
    parser.add_argument(
        "-t", "--tolerance", type=float, default=0.1, help="Tolerance fraction"
    )

    args = parser.parse_args()
    scale = args.scale

    # Get all blocks
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    H, W = image.shape[:2]
    outH, outW = int(scale * H), int(scale * W)

    for i in range(args.num_outputs):
        textureMap = generateTextureMap(
            image, args.block_size, args.overlap, outH, outW, args.tolerance
        )
        textureMap = (255 * textureMap).astype(np.uint8)
        textureMap = cv2.cvtColor(textureMap, cv2.COLOR_RGB2BGR)
        if args.num_outputs == 1:
            cv2.imwrite(args.output_file, textureMap)
            print("Saved output to {}".format(args.output_file))
        else:
            cv2.imwrite(args.output_file.replace(".", "_{}.".format(i)), textureMap)
            print(
                "Saved output to {}".format(
                    args.output_file.replace(".", "_{}.".format(i))
                )
            )
