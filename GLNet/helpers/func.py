import torch
from torch import nn
from ..utils import PhaseMode
from ..models import GLNet

def create_model_load_weights(
    n_class,
    mode=PhaseMode.GlobalOnly,
    gpu_ids=[0],
    evaluation=False,
    path_g=None, path_g2l=None, path_l2g=None
):
    model = GLNet(n_class)
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()

    if (mode is PhaseMode.LocalFromGlobal and not evaluation) or (mode is PhaseMode.GlobalOnly and evaluation):
        # load fixed basic global branch
        partial = torch.load(path_g)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {
            k: v for k, v in partial.items() if k in state and "local" not in k
        }
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    if (mode is PhaseMode.GlobalFromLocal and not evaluation) or (mode is PhaseMode.LocalFromGlobal and evaluation):
        partial = torch.load(path_g2l)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {
            k: v for k, v in partial.items() if k in state
        }  # and "global" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    global_fixed = None
    if mode is PhaseMode.GlobalFromLocal:
        # load fixed basic global branch
        global_fixed = GLNet(n_class)
        global_fixed = nn.DataParallel(global_fixed, device_ids=gpu_ids)
        global_fixed = global_fixed.cuda()
        partial = torch.load(path_g)
        state = global_fixed.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {
            k: v for k, v in partial.items() if k in state and "local" not in k
        }
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        global_fixed.load_state_dict(state)
        global_fixed.eval()

    if mode is PhaseMode.GlobalFromLocal and evaluation:
        partial = torch.load(path_l2g)
        state = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {
            k: v for k, v in partial.items() if k in state
        }  # and "global" not in k}
        # 2. overwrite entries in the existing state dict
        state.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(state)

    if mode is PhaseMode.GlobalOnly or mode is PhaseMode.GlobalFromLocal:
        model.module.backbone_local.eval()
        model.module.fpn_local.eval()
    else:
        model.module.backbone_global.eval()
        model.module.fpn_global.eval()

    return model, global_fixed

def get_optimizer(model, mode=1, learning_rate=2e-5):
    if mode is PhaseMode.GlobalOnly or mode is PhaseMode.GlobalFromLocal:
        # train global
        optimizer = torch.optim.Adam(
            [
                {
                    "params": model.module.backbone_global.parameters(),
                    "lr": learning_rate,
                },
                {"params": model.module.backbone_local.parameters(), "lr": 0},
                {"params": model.module.fpn_global.parameters(), "lr": learning_rate},
                {"params": model.module.fpn_local.parameters(), "lr": 0},
                {
                    "params": model.module.ensemble_conv.parameters(),
                    "lr": learning_rate,
                },
            ],
            weight_decay=5e-4,
        )
    else:
        # train local
        optimizer = torch.optim.Adam(
            [
                {"params": model.module.backbone_global.parameters(), "lr": 0},
                {"params": model.module.backbone_local.parameters(), "lr": learning_rate},
                {"params": model.module.fpn_global.parameters(), "lr": 0},
                {"params": model.module.fpn_local.parameters(), "lr": learning_rate},
                {
                    "params": model.module.ensemble_conv.parameters(),
                    "lr": learning_rate,
                },
            ],
            weight_decay=5e-4,
        )
    return optimizer

def collate(batch):
    image = [b["image"] for b in batch]  # w, h
    label = [b["label"] for b in batch]
    _id = [b["id"] for b in batch]
    return {"image": image, "label": label, "id": _id}

def collate_test(batch):
    image = [b["image"] for b in batch]  # w, h
    _id = [b["id"] for b in batch]
    return {"image": image, "id": _id}
