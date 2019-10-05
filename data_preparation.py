import warnings
import os
from openslide import OpenSlide, deepzoom
from skimage import io
from tqdm import tqdm

DATA_DIR = '/home/hainq/quanghai/Thesis/datasets/paip2019/'
os.mkdir(os.path.join(DATA_DIR, 'training_patches'))

img_pch_dir = os.path.join(DATA_DIR, 'training_patches', 'images')
msk_pch_dir = os.path.join(DATA_DIR, 'training_patches', 'masks')
os.mkdir(img_pch_dir)
os.mkdir(msk_pch_dir)

ids = [f.replace('.svs', '') for f in os.listdir(os.path.join(DATA_DIR, 'training', 'ws_images'))]

for index in ids:
    print(index)
    slide = OpenSlide(os.path.join(DATA_DIR, 'training', 'ws_images', f'{index}.svs'))
    msk = io.imread(os.path.join(DATA_DIR, 'training', 'viable_masks', f'{index}_viable.tif'))
    
    dz = deepzoom.DeepZoomGenerator(slide, tile_size=4000, overlap=48)
    cols, rows = dz.level_tiles[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for row in range(rows):
            for col in tqdm(range(cols)):
                print(row, col)
                tile = dz.get_tile(dz.level_count - 1, (col, row))
                tile.save(os.path.join(img_pch_dir, f'{index}_{row}_{col}.jpg'))

                left, top = dz.get_tile_coordinates(dz.level_count - 1, (col, row))[0]

                cropped_mask = msk[top:(4096 + top), left:(4096 + left)]
                cropped_mask *= 255
                io.imsave(os.path.join(msk_pch_dir, f'{index}_{row}_{col}_mask.jpg'), cropped_mask)
