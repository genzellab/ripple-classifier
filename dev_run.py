
import os
import logging
from argparse import ArgumentParser
import time
from dotenv import load_dotenv

import wandb

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, DeviceStatsMonitor, stochastic_weight_avg
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything, plugins

from data_modules.ripple_module import RippleDataModule
from data_modules.multimodal_ripple_module import MultiModalRippleDataModule
from models.HPCnet import HPCnet
from models.HPC_conformer import HPC_Conformer
from models.PFC_conformer import PFC_Conformer
from models.Multimodal_conformer import MM_Conformer
seed_everything(42)


load_dotenv()
# os.system('wandb login {}'.format(os.getenv('WANDB_API_KEY')))
# time.sleep(10)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


def main(hparams, network):
    # init module
    if hparams.model_load_from_checkpoint:
        model = network.load_from_checkpoint(os.path.join(
            hparams.multimodal_pretrained_folder, hparams.model_file), **vars(hparams))
    else:
        model = network(hparams)
    project_folder = 'ripple_project'
    checkpoint_path = os.path.join(
        './checkpoints/', hparams.model_name)  # "/opt/ml/checkpoints/"

    wandb_logger = WandbLogger(
        name=hparams.model_name, project=project_folder, entity=os.getenv(
            'WANDB_ENTITY'),
        offline=True,)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=hparams.early_stop_num,
        verbose=False,
        mode='min'
    )

    weight_avg = False
    device_stats = DeviceStatsMonitor()
    callbacks = [
        early_stop_callback,
        ModelCheckpoint(
            dirpath=checkpoint_path + "/",
            filename=hparams.model_name + "_{epoch:02d}-{val_loss:.2f}",
            save_last=True,
            save_top_k=4,
            monitor="val_loss",
            mode="min",
            every_n_epochs=1,
        ),
        device_stats,
    ]
    if weight_avg:
        callbacks.append(stochastic_weight_avg.StochasticWeightAveraging())

    trainer = Trainer(
        max_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        num_nodes=hparams.nodes,
        logger=wandb_logger,
        callbacks=callbacks,
        benchmark=bool(hparams.fixed_data),
        precision=hparams.precision,
        default_root_dir=checkpoint_path,
        enable_checkpointing=True,
        strategy=plugins.DDPPlugin(find_unused_parameters=False),  # "ddp",
        auto_select_gpus=True,
        accumulate_grad_batches=hparams.accum_grad_batches,
        gradient_clip_val=hparams.gradient_clip_val,
        accelerator='gpu',
        # profiler='advanced',
        weights_summary='full',
        limit_train_batches=0.4,
        # fast_dev_run=True,
    )

    transforms_comp = None
    if 'multi' in hparams.model_name:
        datamodule = MultiModalRippleDataModule(
            transforms=transforms_comp, num_workers=4, **vars(hparams))
    else:
        datamodule = RippleDataModule(
            transforms=transforms_comp, num_workers=4, **vars(hparams))
    trainer.fit(model, datamodule=datamodule)
    trainer.test(datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    # trainer args
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--model-name', type=str, default='model_debug')
    parser.add_argument('--fixed-data', type=int, default=1,
                        help='if 1, use fixed data can increase the speed of your system if your input sizes dont change.')
    parser.add_argument('--accum_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_val', type=float, default=0.0)
    parser.add_argument("--max_nb_epochs", default=1000, type=int)
    parser.add_argument('--early_stop_num', type=int, default=500)
    parser.add_argument("--batch_size", default=64, type=int)
    # data args
    parser.add_argument('--lazy_load', type=int, default=1)

    # wandb args
    parser.add_argument('--sweep-name', type=str, default="",
                        help='name of the sweep wandb will use to save the results')

    # model args
    parser.add_argument('--data-dir', type=str,
                        default='proc_data/PFC_cmor10_2s', help='path to the data')
    parser.add_argument('--data-dir-HPC', type=str,
                        default='proc_data/HPC_150ms', help='path to the data')
    parser.add_argument('--data-dir-PFC', type=str,
                        default='proc_data/PFC_128ft', help='path to the data')

    parser.add_argument("--num_classes",
                        dest="num_classes",
                        default=3,
                        type=int)
    parser.add_argument("--fold", type=int, default=1)


    #hpc dataset creation args
    parser.add_argument('--wavelet-scales-num', type=int,
                        default=32, help='Wavelet scales num. samples value for linspace')
    parser.add_argument('--data-loc', type=str,
                        default='data/PFCshal', help='File location')
    parser.add_argument('--recording-loc', type=str,
                        default='PFC', help='Recording location')
    parser.add_argument('--wavelet-scales-start', type=int,
                        default=2, help='Wavelet scales start value for linspace')
    parser.add_argument('--wavelet-scales-end', type=int,
                        default=512, help='Wavelet scales end value for linspace')

    parser.add_argument('--wavelet-name', type=str,
                        default='shan', help='Wavelet name')
    parser.add_argument('--wavelet-b', type=float,
                        default=10, help='Wavelet name')
    parser.add_argument('--wavelet-c', type=float,
                        default=1.0, help='Wavelet name')
    parser.add_argument('--event-window-s', type=int,
                        default=2, help='Event window size in seconds')
    parser.add_argument('--sampling-freq', type=float,
                        default=600, help='Sampling frequency')
    parser.add_argument('--output-loc', type=str,
                        default='proc_data/PFC_TMP/', help='Output location')

    # parser.add_argument("--model-type", type=str, default=os.environ['SM_HP_MODEL_TYPE'])
    parser.add_argument("--model-load-from-checkpoint", type=int, default=0)

    network = PFC_Conformer  # PFC_Conformer#HPC_Conformer#HPCnet

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = network.add_model_specific_args(parser)

    # parse params
    print(os.getcwd())

    hparams, _ = parser.parse_known_args()

    print(hparams)
    main(hparams, network)
