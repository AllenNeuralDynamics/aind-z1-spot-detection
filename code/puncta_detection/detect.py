"""
Large-scale puncta detection using single GPU
"""

import logging
import multiprocessing
# from functools import partial
from time import time
from typing import List, Optional, Tuple

import cupy
import numpy as np
import psutil
import torch
import utils
from aind_large_scale_prediction.generator.dataset import create_data_loader
from neuroglancer import CoordinateSpace
from aind_large_scale_prediction.generator.utils import concatenate_lazy_data

# from lazy_deskewing import (create_dispim_config, create_dispim_transform, lazy_deskewing)
from .traditional_detection.puncta_detection_optimized import (
    prune_blobs, traditional_3D_spot_detection)
from .utils.generate_precomputed_format import generate_precomputed_spots
from ._shared.types import ArrayLike

# torch.backends.cudnn.benchmark = False
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def recover_global_position(
    super_chunk_slice: List[slice], internal_slice: List[slice], zyx_points: np.array
) -> np.array:
    """
    Recovers global position of local point
    positions generated due to the chunked
    prediction.

    Parameters
    ----------
    super_chunk_slice: List[slice]
        Super chunk location where the zyx points
        were detected. We divide the entire zarr
        volume in super chunk which then are sent
        to the scheduler to pull internal chunks.

    internal_slice: List[slice]
        List of internal chunks from the super chunks
        that were pulled from the entire zarr volume.
        This list has a one-one correspondence with the
        super chunk list. This means that position 0
        in the super chunk list corresponds to all the
        internal chunks that were served and appear in
        position 0 of this list.

    zyx_points: np.array
        Identified ZYX positions in the entire zarr
        volume.

    Returns
    -------
    np.array
        Array with the global position of the points
    """
    zyx_super_chunk_start = []
    zyx_internal_slice_start = []

    for idx in range(len(internal_slice)):
        zyx_super_chunk_start.append(super_chunk_slice[idx].start)
        zyx_internal_slice_start.append(internal_slice[idx].start)

    zyx_global_points_positions = (
        np.array(zyx_super_chunk_start)
        + np.array(zyx_internal_slice_start)
        + np.array(zyx_points)
    )

    return zyx_global_points_positions


def apply_mask(data: ArrayLike, mask: ArrayLike = None) -> ArrayLike:
    """
    Applies the mask to the current data. This
    should come in the second channel.

    Parameters
    ----------
    data: ArrayLike
        Data to mask.
    
    mask: ArrayLike
        Segmentation mask.

    Returns
    -------
    ArrayLike
        Data after applying masking
    """

    if mask is None:
        return data

    orig_dtype = data.dtype

    mask[ mask > 0 ] = 1
    mask = mask.astype(np.uint8)

    data = data * mask

    return data.astype(orig_dtype)

def z1_puncta_detection(
    dataset_paths: str,
    multiscale: str,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    batch_size: 96,
    output_folder: str,
    logger: logging.Logger,
    super_chunksize: Optional[Tuple[int, ...]] = None,
):
    """
    Chunked puncta detection

    Parameters
    ----------
    dataset_path: str
        Path where the zarr dataset is stored. It could
        be a local path or in a S3 path.

    multiscale: str
        Multiscale to process

    prediction_chunksize: Tuple[int, ...]
        Prediction chunksize the model will pull from
        the raw data

    target_size_mb: int
        Target size in megabytes the data loader will
        load in memory at a time

    n_workers: int
        Number of workers that will concurrently pull
        data from the shared super chunk in memory

    batch_size: int
        Batch size processed each time

    output_folder: str
        Output folder for the detected spots

    logger: logging.Logger
        Logging object

    super_chunksize: Optional[Tuple[int, ...]]
        Super chunk size that will be in memory at a
        time from the raw data. If provided, then
        target_size_mb is ignored. Default: None
    """

    logger.info(f"{20*'='} Running puncta detection {20*'='}")
    logger.info(f"Output folder: {output_folder}")

    utils.print_system_information(logger)

    logger.info(f"Processing dataset {dataset_paths} with mulsticale {multiscale}")

    # Tracking compute resources
    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    """
    # lazy preprocessing
    partial_lazy_deskewing = partial(
        lazy_deskewing,
        multiscale=multiscale,
        camera=camera,
        make_isotropy_voxels=False,
        logger=logger,
    )
    """

    logger.info("Creating chunked data loader")
    shm_memory = psutil.virtual_memory()
    logger.info(f"Shared memory information: {shm_memory}")

    device = None

    pin_memory = True
    if device is not None:
        pin_memory = False
        multiprocessing.set_start_method("spawn", force=True)

    lazy_data = concatenate_lazy_data(
        dataset_paths=dataset_paths,
        multiscale=multiscale,
        concat_axis=-4,  # Concatenation axis
    )

    zarr_data_loader, zarr_dataset = create_data_loader(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        prediction_chunksize=prediction_chunksize,
        n_workers=n_workers,
        batch_size=batch_size,
        dtype=np.float32,  # Allowed data type to process with pytorch cuda
        super_chunksize=super_chunksize,
        lazy_callback_fn=None,  # partial_lazy_deskewing,
        logger=logger,
        device=device,
        pin_memory=pin_memory,
        override_suggested_cpus=False,
        drop_last=True,
        locked_array=False,
    )

    logger.info("Running puncta in chunked data")

    start_time = time()

    # CH 2
    sigma_zyx = [2.0, 1.2, 1.2]
    background_percentage = 25
    pad_size = int(1.6 * max(max(sigma_zyx[1:]), sigma_zyx[0]) * 5)
    min_zyx = [4, 7, 7]
    filt_thresh = 20
    raw_thresh = 30
    context_radius = 3
    radius_confidence = 0.05

    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(f"Number of batches: {total_batches}")
    spots_global_coordinate = None

    for i, sample in enumerate(zarr_data_loader):
        logger.info(
            f"Batch {i}: {sample.batch_tensor.shape} - Pinned?: {sample.batch_tensor.is_pinned()} - dtype: {sample.batch_tensor.dtype} - device: {sample.batch_tensor.device}"
        )

        # sample_cupy = cupy.asarray(sample.batch_tensor)
        #
        # sample_cupy = cupy.from_dlpack(sample.batch_tensor.cuda())

        masked_data = apply_mask(data=sample.batch_tensor, mask=None)

        start_spot_time = time()
        data_block_cupy = cupy.asarray(
            masked_data
        )
        # cupy.from_dlpack(data_block) # [batch_idx, ...]
        same_pointer_in_GPU = False  # sample.batch_tensor.__cuda_array_interface__['data'][0] == data_block_cupy.__cuda_array_interface__['data'][0]
        logger.info(
            f"Same in-GPU memory pointer?: {same_pointer_in_GPU} - Image shape: {data_block_cupy.shape} - cupy dtype: {data_block_cupy.dtype} - Device pulled data: {sample.batch_tensor.device} - Device cupy data: {data_block_cupy.device} - super chunk: {sample.batch_super_chunk} - super chunk slice: {sample.batch_internal_slice}"
        )
        spots = None

        with cupy.cuda.Device(device=device):
            with cupy.cuda.Stream.null:
                for batch_idx in range(0, sample.batch_tensor.shape[0]):
                    logger.info(
                        f"Processing inner batch {batch_idx} out of {sample.batch_tensor.shape[0]} of batch {i}"
                    )
                    # dlpack_sample = to_dlpack(data_block)
                    # Converting to CuPy Array

                    # Making sure CuPy it's running in the correct device
                    spots = traditional_3D_spot_detection(
                        data_block=data_block_cupy[batch_idx, ...],
                        background_percentage=background_percentage,
                        sigma_zyx=sigma_zyx,
                        pad_size=pad_size,
                        min_zyx=min_zyx,
                        filt_thresh=filt_thresh,
                        raw_thresh=raw_thresh,
                        context_radius=context_radius,
                        radius_confidence=radius_confidence,
                        logger=logger,
                    )

                    if spots is None:
                        logger.info(
                            f"No spots found in inner batch {batch_idx} from outer batch {i}"
                        )

                    else:
                        logger.info(
                            f"{len(spots)} spots for in inner batch {batch_idx} from outer batch {i}"
                        )
                        spots_global_coordinate_current = recover_global_position(
                            super_chunk_slice=sample.batch_super_chunk[0],
                            internal_slice=sample.batch_internal_slice[0],
                            zyx_points=spots,
                        )

                        if spots_global_coordinate is None:
                            spots_global_coordinate = (
                                spots_global_coordinate_current.copy()
                            )

                        else:
                            spots_global_coordinate = np.append(
                                spots_global_coordinate,
                                spots_global_coordinate_current,
                                axis=0,
                            )

        end_spot_time = time()
        logger.info(f"Time processing batch {i}: {end_spot_time - start_spot_time}")

        if i + samples_per_iter > total_batches:
            logger.info(
                f"Not enough samples to retrieve from workers, remaining: {i + samples_per_iter - total_batches}"
            )
            break

    end_time = time()

    start_final_prunning_time = time()
    spots_global_coordinate_prunned = prune_blobs(
        blobs_array=spots_global_coordinate, distance=min_zyx[-1] + radius_confidence
    )
    end_final_prunning_time = time()
    logger.info(
        f"Time taken for final prunning {end_final_prunning_time - start_final_prunning_time} before: {len(spots_global_coordinate)} After: {len(spots_global_coordinate_prunned)}"
    )

    # TODO add chunked precomputed format for points with multiscales
    coord_space = CoordinateSpace(
        names=["z", "y", "x"], units=["m", "m", "m"], scales=["5e-7", "5e-7", "5e-7"]
    )
    generate_precomputed_spots(
        spots=spots_global_coordinate_prunned,
        path=f"{output_folder}/precomputed",
        res=coord_space,
    )

    logger.info(f"Processing time: {end_time - start_time} seconds")

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            output_folder,
            "dispim_puncta_detection",
        )
