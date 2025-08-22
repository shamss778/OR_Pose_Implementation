import torch
import os
import shutil
import datetime
from typing import Tuple

def prepare_save_dir(config) -> Tuple[str, bool]:
    """
    Build the save directory path from config, create it if needed, and return it.

    Returns:
        (save_path, created)
    """
    if not getattr(config, "saveX", False):
        return "", False

    run_tag = "{},{},{}epochs,b{},lr{}".format(
        config.model, config.optim, config.epochs, config.batch_size, config.lr
    )
    time_stamp = datetime.datetime.now().strftime("%m-%d-%H:%M")

    save_path = os.path.join(
        config.save_path,      # base
        config.dataName,       # dataset
        time_stamp,          # timestamp
        run_tag,             # run descriptor
    )

    os.makedirs(save_path, exist_ok=True)
    print(f"==> Will save Everything to {save_path}")
    return save_path, True

def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        print('Best Model Saved: ');print(best_path)



