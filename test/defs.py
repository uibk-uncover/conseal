import pathlib

ASSETS_DIR = pathlib.Path('test/assets')
COVER_DIR = ASSETS_DIR / 'cover'
COVER_UG_DIR = COVER_DIR / 'uncompressed_gray'
COVER_UC_DIR = COVER_DIR / 'uncompressed_color'
COVER_CG_DIR = COVER_DIR / 'jpeg_75_gray'
COVER_CC_DIR = COVER_DIR / 'jpeg_75_color'

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
