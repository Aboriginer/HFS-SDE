"""Training and evaluation for score-based generative models. """
import os
import time
import logging
from models import ncsnpp, ddpm
import losses
import sampling
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from utils.utils import *
import utils.datasets as datasets
import tensorflow as tf


FLAGS = flags.FLAGS


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # The directory for saving test results during training
    sample_dir = os.path.join(workdir, "samples_in_train")
    tf.io.gfile.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    tf.io.gfile.makedirs(checkpoint_dir)

    initial_step = int(state["step"])

    # Build pytorch dataloader for training
    train_dl = datasets.get_dataset(config, "training")

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(config)
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(config)
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(config)
    elif config.training.sde.lower() == "hfssde":
        sde = sde_lib.HFS_SDE(config)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        config,
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for epoch in range(config.training.epochs):
        loss_sum = 0
        for step, batch in enumerate(train_dl):
            t0 = time.time()
            k0, csm = batch
            # TODO: mask condition
            label = Emat_xyt_complex(k0, True, csm, 1)  # 1x1x320x320
            label = c2r(label).type(torch.FloatTensor).to(config.device)
            label = scaler(label)

            # Execute one training step
            loss = train_step_fn(state, label)
            loss_sum += loss

            param_num = sum(param.numel() for param in state["model"].parameters())
            # if step % 10 == 0:
            print(
                "Epoch",
                epoch + 1,
                "/",
                config.training.epochs,
                "Step",
                step,
                "loss = ",
                loss.cpu().data.numpy(),
                "loss mean =",
                loss_sum.cpu().data.numpy() / (step + 1),
                "time",
                time.time() - t0,
                "param_num",
                param_num,
            )

        # Save a checkpoint for every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}.pth"), state
            )


def sample(config, workdir):
    """Generate samples.

    Args:
      config: Configuration to use.
      workdir: Working directory.
    """
    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{config.sampling.ckpt}.pth")
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    print("load weights:", ckpt_path)

    if FLAGS.config.sampling.datashift == "head":
        SAMPLING_FOLDER_ID = "_".join(
            [
                FLAGS.config.sampling.acc,
                FLAGS.config.sampling.acs,
                FLAGS.config.sampling.mask_type,
                "ckpt",
                str(config.sampling.ckpt),
                FLAGS.config.sampling.predictor,
                FLAGS.config.training.mean_equal,
                FLAGS.config.sampling.datashift,
                FLAGS.config.sampling.fft,
                str(config.sampling.snr),
                "predictor_mse",
                str(FLAGS.config.sampling.mse),
                "corrector_mse",
                str(FLAGS.config.sampling.corrector_mse),
                str(FLAGS.config.data.centered),
                str(
                    FLAGS.config.sampling.N
                    if FLAGS.config.sampling.accelerated_sampling
                    else ""
                ),
                "seed",
                str(FLAGS.config.seed),
            ]
        )
        test_dl = datasets.get_dataset(
            config, "datashift"
        )  # mode=test:90多张图，modex=sample:一张图，第十张
    elif FLAGS.config.sampling.datashift == "photom":
        SAMPLING_FOLDER_ID = "_".join(
            [
                FLAGS.config.sampling.acc,
                FLAGS.config.sampling.acs,
                FLAGS.config.sampling.mask_type,
                "ckpt",
                str(config.sampling.ckpt),
                FLAGS.config.sampling.predictor,
                FLAGS.config.training.mean_equal,
                FLAGS.config.sampling.datashift,
                FLAGS.config.sampling.fft,
                str(config.sampling.snr),
                "predictor_mse",
                str(FLAGS.config.sampling.mse),
                "corrector_mse",
                str(FLAGS.config.sampling.corrector_mse),
                str(FLAGS.config.data.centered),
                str(
                    FLAGS.config.sampling.N
                    if FLAGS.config.sampling.accelerated_sampling
                    else ""
                ),
                "photom",
                "seed",
                str(FLAGS.config.seed),
            ]
        )
        test_dl = datasets.get_dataset(
            config, "photom"
        )  # mode=test:90多张图，modex=sample:一张图，第十张
    else:
        SAMPLING_FOLDER_ID = "_".join(
            [
                FLAGS.config.sampling.acc,
                FLAGS.config.sampling.acs,
                FLAGS.config.sampling.mask_type,
                "ckpt",
                str(config.sampling.ckpt),
                FLAGS.config.sampling.predictor,
                FLAGS.config.training.mean_equal,
                str(config.sampling.snr),
                "predictor_mse",
                str(FLAGS.config.sampling.mse),
                "corrector_mse",
                str(FLAGS.config.sampling.corrector_mse),
                str(
                    FLAGS.config.data.centered,
                ),
                str(
                    FLAGS.config.sampling.N
                    if FLAGS.config.sampling.accelerated_sampling
                    else ""
                ),
                "--",
                "seed",
                str(FLAGS.config.seed),
            ]
        )
        test_dl = datasets.get_dataset(
            config, "test"
        )  # mode=test:90多张图，modex=sample:一张图，第十张

    FLAGS.config.sampling.folder = os.path.join(FLAGS.workdir, SAMPLING_FOLDER_ID)
    tf.io.gfile.makedirs(FLAGS.config.sampling.folder)

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(config)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == "hfssde":
        sde = sde_lib.HFS_SDE(config)
        sampling_eps = 1e-3  # TODO
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    atb_mask = get_mask(config, "sample")
    train_mask = get_mask(config, "sde")

    # Build the sampling function when sampling is enabled

    sampling_shape = (
        config.sampling.batch_size,
        config.data.num_channels,
        config.data.image_size,
        config.data.image_size,
    )
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, inverse_scaler, sampling_eps, atb_mask, train_mask
    )

    for index, point in enumerate(test_dl):
        print("---------------------------------------------")
        print("---------------- point:", index, "------------------")
        print("---------------------------------------------")

        k0, csm = point
        k0 = k0.to(config.device)
        csm = csm.to(config.device)

        label = Emat_xyt_complex(k0, True, csm, 1.0).to(config.device)

        label_dir = os.path.join("results", FLAGS.config.sampling.datashift)
        if not tf.io.gfile.exists(label_dir):
            tf.io.gfile.makedirs(label_dir)
        save_mat(label_dir, label.to(label), "label", index, normalize=False)

        atb = k0 * atb_mask
        atb_to_image = (
            c2r(Emat_xyt_complex(atb, True, csm, 1))
            .type(torch.FloatTensor)
            .to(config.device)
        )  # 1x2x320x320

        csm = c2r(csm).type(torch.FloatTensor).to(config.device)

        recon, n = sampling_fn(score_model, atb, atb_to_image, csm)

        recon = r2c(recon)

        save_mat(
            FLAGS.config.sampling.folder,
            recon.to(recon),
            "recon",
            index,
            normalize=False,
        )

        from utils.calc import Evaluation_metrics

        ssim, psnr, nmse = Evaluation_metrics(
            label, recon, True if FLAGS.config.sampling.datashift == "photom" else False
        )

        print(
            f"mse_{config.sampling.mse}_snr_{config.sampling.snr}_cmse_{config.sampling.corrector_mse}:"
        )
        print("nmse:", nmse)
        print("ssim:", ssim)
        print("psnr:", psnr)
