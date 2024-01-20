import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf
from datetime import datetime
from utils.utils import setup_seed

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "sample"], "Running mode: train or sample")
flags.mark_flags_as_required(["config", "workdir", "mode"])


def main(argv):
    # set_up seed
    setup_seed(FLAGS.config.seed)
    print(FLAGS.config.seed)
    assert FLAGS.workdir == "results"
    print(FLAGS.config)
    if FLAGS.mode == "train":
        TIMESTAMP = "{0:%Y_%m_%dT%H_%M_%S}".format(datetime.now())
        if FLAGS.config.training.sde == "hfssde":
            MODEL_ID = "_".join(
                [
                    TIMESTAMP,
                    FLAGS.config.model.name,
                    FLAGS.config.training.sde,
                    FLAGS.config.training.mask_type,
                    str(FLAGS.config.training.acs),
                    FLAGS.config.data.normalize_type,
                    str(FLAGS.config.model.beta_max),
                    "N",
                    str(FLAGS.config.model.num_scales),
                ]
            )
        else:
            MODEL_ID = "_".join(
                [
                    TIMESTAMP,
                    FLAGS.config.model.name,
                    FLAGS.config.training.sde,
                    "N",
                    str(FLAGS.config.model.num_scales),
                ]
            )

        FLAGS.workdir = os.path.join(FLAGS.workdir, MODEL_ID)
        tf.io.gfile.makedirs(FLAGS.workdir)

        gfile_stream = open(os.path.join(FLAGS.workdir, "stdout.txt"), "w")
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel("INFO")
        # Run the training pipeline
        run_lib.train(FLAGS.config, FLAGS.workdir)
    elif FLAGS.mode == "sample":
        FLAGS.workdir = os.path.join(FLAGS.workdir, FLAGS.config.sampling.folder)
        run_lib.sample(FLAGS.config, FLAGS.workdir)
    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)
