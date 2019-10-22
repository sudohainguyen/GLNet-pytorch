import os
import numpy as np
import sys
import torch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from GLNet.dataset.paip import Paip, is_image_file, class_to_RGB
# from GLNet.dataset.deep_globe import DeepGlobe, classToRGB, is_image_file
from GLNet.helpers import (
    create_model_load_weights,
    Evaluator,
    collate
)
from GLNet.options import TestingOptions
from GLNet.utils import PhaseMode

torch.backends.cudnn.deterministic = True


def gen_logs(mode, test_scores):
    score_test, score_test_global, score_test_local = test_scores

    log = "================================\n"
    log += mode.name
    log += "\nConfusion Matrix:\n"
    
    log = log + f"Global score: {score_test_global}\n"
    
    if mode is PhaseMode.LocalFromGlobal or mode is PhaseMode.GlobalFromLocal:
        log += f"Local score: {score_test_local}\n"
        log += f"Overall: {score_test}\n"
    return log


def main():
    args = TestingOptions().parse()
    print(args.task_name)

    mode = args.mode
    data_path = args.data_path
    model_path = args.model_path
    batch_size = args.batch_size
    n_class = args.n_class

    ##### sizes are (w, h) #####
    # make sure margin / 32 is over 1.5 AND size_g is divisible by 4
    size_g = (args.size_g, args.size_g)  # resized global image
    size_p = (args.size_p, args.size_p)  # cropped local patch size
    sub_batch_size = args.sub_batch_size  # batch size for train local patches
    
    path_g = os.path.join(model_path, args.path_g)
    path_g2l = os.path.join(model_path, args.path_g2l)
    path_l2g = os.path.join(model_path, args.path_l2g)
    save_predictions = args.save_predictions

    f_log = open(os.path.join(args.log_path, args.task_name + "_test.log"), "w")

    ids_test = [
        image_name
        for image_name in os.listdir(os.path.join(data_path, "test", "images"))
        if is_image_file(image_name)
    ]

    dataset_test = Paip(os.path.join(data_path, "test"), ids_test, label=True, img_shape=4096)
    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        shuffle=False,
        pin_memory=True
    )

    model, global_fixed = create_model_load_weights(
        n_class, mode,
        evaluation=True,
        gpu_ids=args.gpu_ids,
        path_g=path_g, path_g2l=path_g2l, path_l2g=path_l2g
    )
    evaluator = Evaluator(n_class, size_g, size_p, sub_batch_size, mode, False)

    tbar = tqdm(dataloader_test)

    for sample_batched in tbar:
        predictions, predictions_global, _ = evaluator.eval_test(
            sample_batched, model, global_fixed
        )
        score_test, score_test_global, _ = evaluator.get_scores()

        if mode is PhaseMode.GlobalOnly:
            tbar.set_description(
                "global mIoU: %.3f"
                % (np.mean(np.nan_to_num(score_test_global["iou"][1:])))
            )
        else:
            tbar.set_description(
                "agg mIoU: %.3f" % 
                (np.mean(np.nan_to_num(score_test["iou"][1:])))
            )
        images = sample_batched["image"]
        labels = sample_batched["label"]

        if save_predictions:
            if not os.path.exists(save_predictions):
                os.mkdir(save_predictions)
            for i in range(len(images)):
                if mode is PhaseMode.GlobalOnly:
                    image = Image.fromarray(
                        np.array(predictions_global[i], dtype=np.uint8) * 255
                    )
                else:
                    image = Image.fromarray(
                        np.array(predictions[i], dtype=np.uint8) * 255
                    )
                Image.fromarray(np.array(labels[i], dtype=np.uint8) * 255)\
                    .save(os.path.join(save_predictions, f"{sample_batched['id'][i]}_mask.png"))
                image.save(os.path.join(save_predictions, f"{sample_batched['id'][i]}_pred.png"))
        
    test_scores = evaluator.get_scores()
    evaluator.reset_metrics()

    log = gen_logs(mode, test_scores)
    print(log)
    f_log.write(log)
    f_log.flush()
    f_log.close()


if __name__ == "__main__":
    main()
