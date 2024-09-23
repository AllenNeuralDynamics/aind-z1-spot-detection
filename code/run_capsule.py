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
    output_folder = RESULTS_FOLDER
    utils.create_folder(dest_dir=str(output_folder), verbose=True)

    # Data
    IMAGE_PATH = "HCR_736207-01_2024-08-08_13-00-00/fused/channel_638.zarr"

    DATA_PATH = f"{DATA_FOLDER}/{IMAGE_PATH}"
    SEGMENTATION_PATH = f"{DATA_FOLDER}/HCR_736207-01_2024-08-08_13-00-00_upsampled-masks/segmentation_mask.zarr"
    # If using the bucket path directly, provide credentials to the capsule
    # f"s3://{BUCKET_NAME}/{IMAGE_PATH}"

    logger = utils.create_logger(output_log_path=str(output_folder))

    # Puncta detection parameters

    sigma_zyx = [1.8, 1.0, 1.0]
    background_percentage = 25
    axis_pad = int(1.6 * max(max(sigma_zyx[1:]), sigma_zyx[0]) * 5)
    min_zyx = [3, 3, 3]
    filt_thresh = 20
    raw_thresh = 180
    context_radius = 3
    radius_confidence = 0.05

    # Data loader params
    puncta_params = {
        "dataset_path": DATA_PATH,
        "segmentation_mask_path": SEGMENTATION_PATH,
        "multiscale": "0",
        "prediction_chunksize": (128, 128, 128),
        "target_size_mb": 3048,
        "n_workers": 0,
        "batch_size": 1,
        "axis_pad": axis_pad,
        "output_folder": output_folder,
        "logger": logger,
        "super_chunksize": None,
        "spot_parameters": {
            "sigma_zyx": sigma_zyx,
            "background_percentage": background_percentage,
            "min_zyx": min_zyx,
            "filt_thresh": filt_thresh,
            "raw_thresh": raw_thresh,
            "context_radius": context_radius,
            "radius_confidence": radius_confidence,
        },
    }

    logger.info(
        f"Dataset path: {puncta_params['dataset_path']} - Puncta detection params: {puncta_params}"
    )

    z1_puncta_detection(**puncta_params)


if __name__ == "__main__":
    # cProfile.run('main()', filename="/results/compute_costs.dat")
    run()
