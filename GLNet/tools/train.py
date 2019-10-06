import os
import sys
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from GLNet.dataset.deep_globe import DeepGlobe, classToRGB, is_image_file
from GLNet.utils.loss import FocalLoss
from GLNet.utils.lovasz_losses import lovasz_softmax
from GLNet.utils.lr_scheduler import LR_Scheduler
from GLNet.helper import (
    create_model_load_weights,
    get_optimizer,
    Trainer,
    Evaluator,
    collate
)
from GLNet.options import TrainingOptions
from GLNet.utils import PhaseMode

torch.backends.cudnn.deterministic = True

def main():
    args = TrainingOptions().parse()
    print(args.task_name)

    mode = args.mode
    data_path = args.data_path
    model_path = args.model_path
    num_epochs = args.num_epochs
    learning_rate = args.lr
    lamb_fmreg = args.lamb_fmreg
    batch_size = args.batch_size
    n_class = args.n_class

    print("mode:", mode.name)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)


    ids_train = [
        image_name
        for image_name in os.listdir(os.path.join(data_path, "train", "images"))
        if is_image_file(image_name)
    ]

    # if args.validation:
    ids_val = [
        image_name
        for image_name in os.listdir(os.path.join(data_path, "val", "images"))
        if is_image_file(image_name)
    ]

    dataset_train = DeepGlobe(os.path.join(data_path, "train"), ids_train, label=True, transform=True)
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        shuffle=True,
        pin_memory=True,
    )

    dataset_val = DeepGlobe(os.path.join(data_path, "val"), ids_val, label=True)
    dataloader_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate,
        shuffle=False,
        pin_memory=True,
    )

    ##### sizes are (w, h) #####
    # make sure margin / 32 is over 1.5 AND size_g is divisible by 4
    size_g = (args.size_g, args.size_g)  # resized global image
    size_p = (args.size_p, args.size_p)  # cropped local patch size
    sub_batch_size = args.sub_batch_size  # batch size for train local patches

    print("creating models......")
    path_g = os.path.join(model_path, args.path_g)
    path_g2l = os.path.join(model_path, args.path_g2l)
    path_l2g = os.path.join(model_path, args.path_l2g)

    model, global_fixed = create_model_load_weights(n_class, mode, False, \
        path_g=path_g, path_g2l=path_g2l, path_l2g=path_l2g)

    optimizer = get_optimizer(model, mode, learning_rate=learning_rate)
    scheduler = LR_Scheduler("poly", learning_rate, num_epochs, len(dataloader_train))
    criterion = lambda x, y: 0.5 * FocalLoss(x, y) + 0.5 * lovasz_softmax(x, y)
    
    writer = SummaryWriter(log_dir=args.log_path + args.task_name)
    f_log = open(args.log_path + args.task_name + ".log", "w")

    trainer = Trainer(
        criterion, optimizer, n_class,
        size_g, size_p,
        sub_batch_size, mode, lamb_fmreg
    )
    
    # if args.validation:
    evaluator = Evaluator(n_class, size_g, size_p, sub_batch_size, mode)

    best_pred = 0.0
    print("start training......")

    for epoch in range(num_epochs):
        trainer.set_train(model)
        optimizer.zero_grad()
        tbar = tqdm(dataloader_train)
        train_loss = 0
        for i_batch, sample_batched in enumerate(tbar):
            scheduler(optimizer, i_batch, epoch, best_pred)
            loss = trainer.train(sample_batched, model, global_fixed)
            train_loss += loss.item()
            score_train, score_train_global, score_train_local = trainer.get_scores()
            
            cur_loss = train_loss / (i_batch + 1)
            if mode is PhaseMode.GlobalOnly:
                tbar.set_description(
                    "Train loss: %.3f; global mIoU: %.3f"
                    % (
                        cur_loss,
                        np.mean(np.nan_to_num(score_train_global["iou"]))
                    )
                )
            else:
                tbar.set_description(
                    "Train loss: %.3f; agg mIoU: %.3f"
                    % (
                        cur_loss,
                        np.mean(np.nan_to_num(score_train["iou"]))
                    )
                )

        score_train, score_train_global, score_train_local = trainer.get_scores()
        trainer.reset_metrics()

        if epoch % args.validation_steps == 0:
            # if args.validation:
            with torch.no_grad():
                model.eval()
                print("Validating...")

                tbar = tqdm(dataloader_val)

                for i_batch, sample_batched in enumerate(tbar):
                    predictions, predictions_global, predictions_local = evaluator.eval_test(
                        sample_batched, model, global_fixed
                    )
                    score_val, score_val_global, score_val_local = evaluator.get_scores()

                    val_acc = 0.0
                    if mode is PhaseMode.GlobalOnly:
                        tbar.set_description(
                            "global mIoU: %.3f"
                            % (np.mean(np.nan_to_num(score_val_global["iou"])))
                        )
                        val_acc = np.mean(np.nan_to_num(score_val_global["iou"]))
                    else:
                        tbar.set_description(
                            "agg mIoU: %.3f" % (np.mean(np.nan_to_num(score_val["iou"])))
                        )
                        val_acc = np.mean(np.nan_to_num(score_val["iou"]))

                    # Only save best model which have highest validation accuracy
                    if val_acc > best_pred:
                        best_pred = val_acc
                        torch.save(model.state_dict(), "./saved_models/" + args.task_name + ".pth")

                    images = sample_batched["image"]
                    labels = sample_batched["label"]  # PIL images

                    if i_batch * batch_size + len(images) > (epoch % len(dataloader_val)) \
                        and i_batch * batch_size <= (epoch % len(dataloader_val)):

                        writer.add_image(
                            "image",
                            transforms.ToTensor()(images[(epoch % len(dataloader_val)) - i_batch * batch_size]),
                            epoch,
                        )
                        writer.add_image(
                            "mask",
                            classToRGB(np.array(labels[(epoch % len(dataloader_val)) - i_batch * batch_size])) * 255.0,
                            epoch,
                        )
                        writer.add_image(
                            "prediction_global",
                            classToRGB(predictions_global[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255.0,
                            epoch,
                        )

                        if mode is PhaseMode.LocalFromGlobal or mode is PhaseMode.GlobalFromLocal:
                            writer.add_image(
                                "prediction_local",
                                classToRGB(predictions_local[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255.0,
                                epoch,
                            )
                            writer.add_image(
                                "prediction",
                                classToRGB(predictions[(epoch % len(dataloader_val)) - i_batch * batch_size]) * 255.0,
                                epoch,
                            )

                log = (
                    "epoch [{}/{}] IoU: train = {:.4f}, val = {:.4f}\n".format(
                        epoch + 1,
                        num_epochs,
                        np.mean(np.nan_to_num(score_train["iou"])),
                        np.mean(np.nan_to_num(score_val["iou"])),
                    )
                )
                log = (
                    log
                    + "epoch [{}/{}] Local  -- IoU: train = {:.4f}, val = {:.4f}\n".format(
                        epoch + 1,
                        num_epochs,
                        np.mean(np.nan_to_num(score_train_local["iou"])),
                        np.mean(np.nan_to_num(score_val_local["iou"])),
                    )
                )
                log = (
                    log
                    + "epoch [{}/{}] Global -- IoU: train = {:.4f}, val = {:.4f}\n".format(
                        epoch + 1,
                        num_epochs,
                        np.mean(np.nan_to_num(score_train_global["iou"])),
                        np.mean(np.nan_to_num(score_val_global["iou"]))
                    )
                )

                log = log + f"Train: {score_train['iou']}\n"
                log = log + f"Local train: {score_train_local['iou']}"
                log = log + f"Global train: {score_train_global['iou']}"
                # if args.validation:
                log = log + f"Val: {score_val['iou']}\n"
                log = log + f"Local val: {score_val_local['iou']}"
                log = log + f"Global val: {score_val_global['iou']}"

                
                log += "================================\n"
                print(log)
                f_log.write(log)
                f_log.flush()
                f_log.close()

    
if __name__ == "__main__":
    main()
