import sys
import time
import datetime
import zarr
from math import ceil
from copy import deepcopy
from typing import Dict
from collections import Counter
from typing import Callable
import yaml
import gc
import os
from pathlib import Path
from ignite.engine import Engine
import numpy as np
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", help="Model name")
parser.add_argument("-d", "--debug", help="Debug mode", action='store_true')
parser.add_argument("-gpu", "--gpu", help="Training on GPU", action='store_true')
args = parser.parse_args()

# ======================== pytorch ========================
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset, Dataset
from torchvision.models import resnet18, wide_resnet50_2, resnet34
from pytorch_pfn_extras.training import extension, trigger, IgniteExtensionsManager
from pytorch_pfn_extras.training.extensions import util
from pytorch_pfn_extras.training.extensions import log_report as log_report_module
from pytorch_pfn_extras.training.extensions.print_report import PrintReport
from pytorch_pfn_extras.training.triggers import MinValueTrigger
import pytorch_pfn_extras.training.extensions as E
import pytorch_pfn_extras as ppe

# ======================== l5kit ========================
import l5kit
print("l5kit version:", l5kit.__version__)
from l5kit.evaluation import create_chopped_dataset
from l5kit.rasterization import build_rasterizer
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.data import ChunkedDataset, LocalDataManager

# ======================== models ========================
from model.baseline import MultiModal
from model.resnet18_gru import Resnet18GRU


# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet34',
        'history_num_frames': 20,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },
    'raster_params': {
        'raster_size': [512, 512],
        'pixel_size': [0.2, 0.2],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4
    },
    'valid_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },
    'train_params': {
        'max_num_steps': 20000,
        'checkpoint_every_n_steps': 5000,
    }
}

flags_dict = {
    "debug": args.debug,
    # --- Data configs ---
    "l5kit_data_folder": "./dataset/",
    # --- Model configs ---
    "pred_mode": "multi",
    # --- Training configs ---
    "device": "cuda:0" if args.gpu else "cpu",  # change this to 'cuda:0' if put on server
    "out_dir": "results",
    "epoch": 2,
    "snapshot_freq": 100,
}


# ==================================================================================== #
#                                   Dataset Utils                                      #
# ==================================================================================== #
class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        batch = self.dataset[index]
        return self.transform(batch)

    def __len__(self):
        return len(self.dataset)
    
def transform(batch):
    """Split Batch into different segments

    Args:
        batch (Tensor): batch tensor with integrated segments

    Returns:
        Tuple(Tensor): split batch into image, target_positions, target_availabilities
    """
    return batch["image"], batch["target_positions"], batch["target_availabilities"]

class DotDict(dict):
    """dot.notation access to dictionary attributes
    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



# ==================================================================================== #
#                                  Training Utils                                      #
# ==================================================================================== #
def save_yaml(filepath, content, width=120):
    with open(filepath, 'w') as f:
        yaml.dump(content, f, width=width)

        
def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(
        pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len,
                        num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (
        batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones(
        (batch_size,))), "confidences should sum to 1"
    assert avails.shape == (
        batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(
    ), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    # reduce coords and use availability
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)
    # when confidence is 0 log goes to -inf, but we're fine with it
    with np.errstate(divide="ignore"):
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * \
            torch.sum(error, dim=-1)  # reduce time
    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    # error are negative at this point, so max() gives the minimum one
    max_value, _ = error.max(dim=1, keepdim=True)
    error = -torch.log(torch.sum(torch.exp(error - max_value),
                                 dim=-1, keepdim=True)) - max_value  # reduce modes
    return torch.mean(error)


def eval_func(*batch):
    loss, metrics = model(*[elem.to(device) for elem in batch])

    
def create_trainer(model, optimizer, device) -> Engine:
    model.to(device)

    def update_fn(engine, batch):
        model.train()
        optimizer.zero_grad()
        loss, metrics = model(*[elem.to(device) for elem in batch])
        loss.backward()
        optimizer.step()
        return metrics
    trainer = Engine(update_fn)
    return trainer
    
    

# ==================================================================================== #
#                                  Training Utils                                      #
# ==================================================================================== #
# class LyftMultiModel(nn.Module):
#     def __init__(self, cfg: Dict, num_modes=3):
#         super().__init__()
#         # TODO: support other than resnet18?
#         backbone = resnet18(pretrained=True, progress=True)
#         self.backbone = backbone
#         num_history_channels = (
#             cfg["model_params"]["history_num_frames"] + 1) * 2
#         num_in_channels = 3 + num_history_channels
#         self.backbone.conv1 = nn.Conv2d(
#             num_in_channels,
#             self.backbone.conv1.out_channels,
#             kernel_size=self.backbone.conv1.kernel_size,
#             stride=self.backbone.conv1.stride,
#             padding=self.backbone.conv1.padding,
#             bias=False,
#         )
#         # This is 512 for resnet18 and resnet34;
#         # And it is 2048 for the other resnets
#         backbone_out_features = 512
#         # X, Y coords for the future positions (output shape: Bx50x2)
#         self.future_len = cfg["model_params"]["future_num_frames"]
#         num_targets = 2 * self.future_len
#         # You can add more layers here.
#         self.head = nn.Sequential(
#             # nn.Dropout(0.2),
#             nn.Linear(in_features=backbone_out_features, out_features=4096),
#         )
#         self.num_preds = num_targets * num_modes
#         self.num_modes = num_modes
#         self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

#     def forward(self, x):
#         x = self.backbone.conv1(x)
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)

#         x = self.backbone.layer1(x)
#         x = self.backbone.layer2(x)
#         x = self.backbone.layer3(x)
#         x = self.backbone.layer4(x)

#         x = self.backbone.avgpool(x)
#         x = torch.flatten(x, 1)

#         x = self.head(x)
#         x = self.logit(x)

#         # pred (bs)x(modes)x(time)x(2D coords)
#         # confidences (bs)x(modes)
#         bs, _ = x.shape
#         pred, confidences = torch.split(x, self.num_preds, dim=1)
#         pred = pred.view(bs, self.num_modes, self.future_len, 2)
#         assert confidences.shape == (bs, self.num_modes)
#         confidences = torch.softmax(confidences, dim=1)
#         return pred, confidences

class ModelNotFoundException(Exception):
        def __init__(self, message):
            super().__init__(message)


def load_model(model):
    if model == 'baseline':
        return MultiModal(cfg)
    if model == 'resnet18_gru':
        return Resnet18GRU(cfg, device)
    raise ModelNotFoundException("ModelNotFoundException: Unable to find the model: {}".format(model))


class MultiRegressor(nn.Module):
    """Multi mode prediction"""

    def __init__(self, predictor, lossfun=pytorch_neg_multi_log_likelihood_batch):
        super().__init__()
        self.predictor = predictor
        self.lossfun = lossfun

    def forward(self, image, targets, target_availabilities):
        pred, confidences = self.predictor(image)
        loss = self.lossfun(targets, pred, confidences, target_availabilities)
        metrics = {
            "loss": loss.item(),
            "nll": pytorch_neg_multi_log_likelihood_batch(targets, pred, confidences, target_availabilities).item()
        }
        ppe.reporting.report(metrics, self)
        return loss, metrics


if __name__ == "__main__":
    # ============================= Load Config ============================= 
    flags = DotDict(flags_dict)
    out_dir = Path(flags.out_dir)
    os.makedirs(str(out_dir), exist_ok=True)
    save_yaml(out_dir / 'flags.yaml', flags_dict)
    save_yaml(out_dir / 'cfg.yaml', cfg)
    debug = flags.debug
    # move model to GPU/CPU
    device = torch.device(flags.device)

    try:
        predictor = load_model(args.model)
    except ModelNotFoundException as e:
        print(e)
        os._exit(0)
    model = MultiRegressor(predictor)    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ============================= Load Dataset ============================= 
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = flags.l5kit_data_folder
    dm = LocalDataManager(None)

    print('='*10 + 'Loading Training Data' + '='*10)
    train_cfg = cfg["train_data_loader"]
    # Rasterizer
    rasterizer = build_rasterizer(cfg, dm)
    # Train dataset/dataloader
    train_path = "scenes/sample.zarr" if debug else train_cfg["key"]
    train_zarr = ChunkedDataset(dm.require(train_path)).open()
    train_agent_dataset = AgentDataset(cfg, train_zarr, rasterizer)
    train_dataset = TransformDataset(train_agent_dataset, transform)
    if debug:
        # Only use subset dataset for fast check...
        train_dataset = Subset(
            train_dataset, 
            np.arange(cfg["train_data_loader"]["batch_size"] * 40)
            )
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_cfg["shuffle"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"]
        )
    print(train_agent_dataset)
    
    # GENERATE AND LOAD CHOPPED DATASET
    print('='*10 + 'Loading Validation' + '='*10)
    valid_cfg = cfg["valid_data_loader"]
    valid_path = "scenes/sample.zarr" if debug else valid_cfg["key"]
    num_frames_to_chop = 100
    MIN_FUTURE_STEPS = 10
    valid_base_path = create_chopped_dataset(
        dm.require(valid_path), 
        cfg["raster_params"]["filter_agents_threshold"],
        num_frames_to_chop, 
        cfg["model_params"]["future_num_frames"], 
        MIN_FUTURE_STEPS
        )
    valid_zarr_path = str(Path(valid_base_path) /
                          Path(dm.require(valid_path)).name)
    valid_mask_path = str(Path(valid_base_path) / "mask.npz")
    valid_gt_path = str(Path(valid_base_path) / "gt.csv")
    valid_zarr = ChunkedDataset(valid_zarr_path).open()
    valid_mask = np.load(valid_mask_path)["arr_0"]
    # ===== INIT DATASET AND LOAD MASK
    valid_agent_dataset = AgentDataset(
        cfg, valid_zarr, rasterizer, agents_mask=valid_mask)
    valid_dataset = TransformDataset(valid_agent_dataset, transform)
    valid_loader = DataLoader(valid_dataset,
                              shuffle=valid_cfg["shuffle"],
                              batch_size=valid_cfg["batch_size"],
                              num_workers=valid_cfg["num_workers"])
    print(valid_agent_dataset)
    print("# AgentDataset train:", len(train_agent_dataset),
          "#valid", len(valid_agent_dataset))
    print("# ActualDataset train:", len(
        train_dataset), "#valid", len(valid_dataset))

    # ============================= Training Setup ============================= 
    trainer = create_trainer(model, optimizer, device)
    valid_evaluator = E.Evaluator(
        valid_loader,
        model,
        progress_bar=True,
        eval_func=eval_func,
    )
    log_trigger = (10 if debug else 100, "iteration")
    log_report = E.LogReport(trigger=log_trigger)
    extensions = [
        log_report,  # Save `log` to file
        valid_evaluator,  # Run evaluation for valid dataset in each epoch.
        # E.FailOnNonNumber()  # Stop training when nan is detected.
    ]
    extensions.extend([
        # Show progress bar during training
        E.ProgressBar(update_interval=10 if debug else 100),
        E.PrintReport(),  # Print "log" to terminal
    ])
    epoch = flags.epoch
    models = {"main": model}
    optimizers = {"main": optimizer}
    manager = IgniteExtensionsManager(
        trainer,
        models,
        optimizers,
        epoch,
        extensions=extensions,
        out_dir=str(out_dir),
    )
    # Save predictor.pt every snapshot
    # manager.extend(E.snapshot_object(predictor, "predictor_{.updater.iteration}.pt"),
    #                trigger=(flags.snapshot_freq, "iteration"))
    manager.extend(E.snapshot_object(predictor, "predictor.pt"),
                   trigger=(flags.snapshot_freq, "iteration"))
    # --- lr scheduler ---
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)
    manager.extend(lambda manager: scheduler.step(), trigger=(1, "iteration"))
    manager.extend(E.observe_lr(optimizer=optimizer), trigger=log_trigger)
    
    # ============================= Start Training ============================= 
    print('='*10 + ' Start Training ... ' + '='*10)
    trainer.run(train_loader, max_epochs=epoch)
