import os
import warnings
import multiprocessing
from time import time
from shutil import rmtree
from skimage import io
from openslide import OpenSlide, deepzoom
from tqdm import tqdm


def extract_svs(process, start_index, end_index):
    for f in paths[start_index : end_index]:
        fname = os.path.basename(f)[:-4]
        target_dir = traindir
        if fname in val_idx:
            target_dir = valdir
        elif fname in test_idx:
            target_dir = testdir

        slide = OpenSlide(f)
        dz = deepzoom.DeepZoomGenerator(slide, tile_size=4096, overlap=0)
        msk = io.imread(os.path.join(ORG_DIR, 'viable_masks', f'{fname}_viable.tif'))
        cols, rows = dz.level_tiles[-1] 
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for row in range(rows):
                for col in range(cols):
                    tile = dz.get_tile(dz.level_count - 1, (col, row))
                    left, top = dz.get_tile_coordinates(dz.level_count - 1, (col, row))[0] 
                    cropped_mask = msk[top:(4096 + top), left:(4096 + left)]
                    tile.save(os.path.join(target_dir, 'images', f'{fname}_{row}_{col}.jpg'))
                    io.imsave(os.path.join(target_dir, 'masks', f'{fname}_{row}_{col}_mask.tif'), cropped_mask)
        slide.close()
        print(f'Process #{process} extracted {f}')
    return f'Process #{process} finished'


if __name__ == "__main__":
    DATA_DIR = '/home/hainq/quanghai/Thesis/datasets/paip2019/'
    ORG_DIR = os.path.join(DATA_DIR, 'training')

    if os.path.exists(os.path.join(DATA_DIR, 'training_tiles')):
        rmtree(os.path.join(DATA_DIR, 'training_tiles'))
    os.mkdir(os.path.join(DATA_DIR, 'training_tiles'))

    traindir = os.path.join(DATA_DIR, 'training_tiles', 'train')
    valdir = os.path.join(DATA_DIR, 'training_tiles', 'val')
    testdir = os.path.join(DATA_DIR, 'training_tiles', 'test')

    os.mkdir(traindir)
    os.mkdir(os.path.join(traindir, 'images'))
    os.mkdir(os.path.join(traindir, 'masks'))
    os.mkdir(valdir)
    os.mkdir(os.path.join(valdir, 'images'))
    os.mkdir(os.path.join(valdir, 'masks'))
    os.mkdir(testdir)
    os.mkdir(os.path.join(testdir, 'images'))
    os.mkdir(os.path.join(testdir, 'masks'))

    val_idx = ['01_01_0110', '01_01_0122', '01_01_0139']
    test_idx = ['01_01_0087', '01_01_0100', '01_01_0115', '01_01_0130', '01_01_0136']
    paths = [os.path.join(ORG_DIR, 'ws_images', f) for f in os.listdir(os.path.join(ORG_DIR, 'ws_images'))]

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)
    num_wsi = len(paths)

    images_per_process = num_wsi // num_processes

    tasks = []

    for process in range(num_processes):
        start_index = process * images_per_process
        end_index = (process + 1) * images_per_process
        next_index = (process + 2) * images_per_process + 1
        if next_index + 1 > len(paths):
            end_index = len(paths) - 1
        tasks.append((process, start_index, end_index))
        print(
            "Task #"
            + str(process)
            + ": Process slides "
            + str(start_index)
            + " to "
            + str(end_index)
        )
    t = time()
    results = []
    for task in tasks:
        results.append(pool.apply_async(extract_svs, task))

    for res in results:
        status = res.get()
        print(status)

    print(time() - t)
