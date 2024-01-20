import h5py
import utils.datasets as datasets
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "sample"], "Running mode: train or sample")
flags.mark_flags_as_required(["config", "workdir", "mode"])


def main(argv):
    test_dl = datasets.get_dataset(FLAGS.config, "test")

    for index, point in enumerate(test_dl):
        print("---------------------------------------------")
        print("---------------- point:", index, "------------------")
        print("---------------------------------------------")
        k0, csm = point
        if index == 9:
            h_kspce = h5py.File(
                "/data0/chentao/data/fastMRI_knee_sample/T1_data/T1_sample.h5", "w"
            )
            h_csm = h5py.File(
                "/data0/chentao/data/fastMRI_knee_sample/output_maps/T1_sample.h5", "w"
            )
            h_kspce.close()
            h_csm.close()


if __name__ == "__main__":
    app.run(main)
