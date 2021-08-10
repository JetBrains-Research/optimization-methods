import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data_module.jsonl_data_module import JsonlDataModule
from models import CodeGNNGRU, LeClairCodeGNNGRU
from utils.callbacks import UploadCheckpointCallback, PrintEpochResultCallback
from utils.common import filter_warnings, print_config

@hydra.main(config_path="configs", config_name="codegnn_codexglue_java")
def train_codegnn(config: DictConfig):
    filter_warnings()
    seed_everything(config.seed)

    print_config(config, ["hydra", "log_offline"])

    data_module = JsonlDataModule(config)
    data_module.prepare_data()
    data_module.setup()
    model: LightningModule
    if config.model_type == 'codegnngru':
        model = CodeGNNGRU(config, data_module.vocabulary)
    elif config.model_type == 'codegnngru_leclair':
        model = LeClairCodeGNNGRU(config, data_module.vocabulary)
    else:
        raise NotImplementedError('No such model')

    # define logger
    wandb_logger = WandbLogger(project=f"codegnn-{config.dataset}", log_model=False, offline=config.log_offline)
    wandb_logger.watch(model)
    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        period=config.save_every_epoch,
        save_top_k=-1,
    )
    upload_checkpoint_callback = UploadCheckpointCallback(wandb_logger.experiment.dir)
    # define early stopping callback
    early_stopping_callback = EarlyStopping(patience=config.hyper_parameters.patience, monitor="val_loss", verbose=True,
                                            mode="min")
    # define callback for printing intermediate result
    print_epoch_result_callback = PrintEpochResultCallback("train", "val")
    # use gpu if it exists
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateMonitor("step")

    trainer = Trainer(
        max_epochs=config.hyper_parameters.n_epochs,
        gradient_clip_val=config.hyper_parameters.clip_norm,
        deterministic=True,
        check_val_every_n_epoch=config.val_every_epoch,
        log_every_n_steps=config.log_every_step,
        logger=wandb_logger,
        gpus=gpu,
        progress_bar_refresh_rate=config.progress_bar_refresh_rate,
        callbacks=[
            lr_logger,
            early_stopping_callback,
            checkpoint_callback,
            upload_checkpoint_callback,
            print_epoch_result_callback,
        ],
        resume_from_checkpoint=config.resume_checkpoint,
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test()


if __name__ == "__main__":
    train_codegnn()
