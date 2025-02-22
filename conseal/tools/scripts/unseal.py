
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def run():
    parser = argparse.ArgumentParser(description='Analyzes a cover-stego pair of images')
    parser.add_argument('image1', type=str, help='First image')
    parser.add_argument('image2', type=str, help='Second image')
    parser.add_argument('--diff',
        const=True,
        default=False,
        action='store',
        nargs='?',
        help='Whether to show the difference image')
    args = parser.parse_args()

    # config
    np.set_printoptions(precision=3)

    # load and compare
    x1 = np.array(Image.open(args.image1))
    x2 = np.array(Image.open(args.image2))
    assert x1.shape == x2.shape, 'images must be of the same shape'

    #

    delta = x1.astype('int32') - x2.astype('int32')
    changes, counts = np.unique(delta.flatten(), return_counts=True)
    print('====================================================')
    print(f'Change rate:', round((x1 != x2).mean(), 4))
    print('Changes:', {k: round(v, 4) for k, v in zip(changes, counts / x1.size)})
    print('====================================================')

    if args.diff is not False:
        d = x1.copy()
        d[x1 != x2] = 255
        d_im = Image.fromarray(d)
        if isinstance(args.diff, str):
            d_im.save(args.diff)
            print(f'Difference image saved to {args.diff}.')
        else:
            d_im.show()
            # print(f'Showing the difference image on screen')