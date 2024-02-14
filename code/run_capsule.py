"""
Scripts that runs the Code Ocean capsule
"""

import os
from pathlib import Path

from puncta_detection.detect import z1_puncta_detection
from puncta_detection.utils import utils


def run():
    """
    Run function
    """

    # Code Ocean folders
    RESULTS_FOLDER = Path(os.path.abspath("../results"))
    # SCRATCH_FOLDER = Path(os.path.abspath("../scratch"))
    DATA_FOLDER = Path(os.path.abspath("../data"))

    # Output folder
    output_folder = RESULTS_FOLDER.joinpath("puncta_detection_results")
    utils.create_folder(dest_dir=str(output_folder), verbose=True)

    # Data
    # BUCKET_NAME = "aind-open-data"
    # IMAGE_PATH = "HCR_BL6-000_2023-06-1_00-00-00_fused_2024-02-09_13-28-49/channel_2.zarr"
    IMAGE_PATH = (
        "HCR_BL6-000_2023-06-07_00-00-00_fused_2024-02-09_15-52-18/channel_2.zarr"
    )

    DATA_PATH = f"{DATA_FOLDER}/{IMAGE_PATH}"
    # If using the bucket path directly, provide credentials to the capsule
    # f"s3://{BUCKET_NAME}/{IMAGE_PATH}"

    logger = utils.create_logger(output_log_path=str(output_folder))

    # Data loader params
    puncta_params = {
        "dataset_path": DATA_PATH,
        "multiscale": "0",
        "prediction_chunksize": (128, 128, 128),  # (256, 256, 256),
        "target_size_mb": 1024,  # 2048,#3072,
        "n_workers": 16,
        "batch_size": 1,
        "output_folder": output_folder,
        "logger": logger,
        "super_chunksize": None,
    }

    co_cpus = int(utils.get_code_ocean_cpu_limit())

    if puncta_params.get("n_workers") > co_cpus:
        raise ValueError(
            f"Provided workers {puncta_params.get('n_workers')} > current workers {co_cpus}"
        )

    logger.info(f"Puncta detection params: {puncta_params}")

    z1_puncta_detection(**puncta_params)


if __name__ == "__main__":
    # cProfile.run('main()', filename="/results/compute_costs.dat")
    run()
