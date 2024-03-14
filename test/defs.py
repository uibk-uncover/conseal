import pathlib

ASSETS_DIR = pathlib.Path('test/assets')
COVER_DIR = ASSETS_DIR / 'cover'
COVER_UNCOMPRESSED_GRAY_DIR = COVER_DIR / 'uncompressed_gray'
COVER_UNCOMPRESSED_COLOR_DIR = COVER_DIR / 'uncompressed_color'
COVER_COMPRESSED_GRAY_DIR = COVER_DIR / 'jpeg_75_gray'
COVER_COMPRESSED_COLOR_DIR = COVER_DIR / 'jpeg_75_color'

TEST_IMAGES = [
    'seal1',
    'seal2',
    'seal3',
    'seal4',
    'seal5',
    'seal6',
    'seal7',
    'seal8',
]
