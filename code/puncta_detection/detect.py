"""
Large-scale puncta detection using single GPU
"""

import logging
import multiprocessing
import os
# from functools import partial
from time import time
from typing import Dict, List, Optional, Tuple

import cupy
import numpy as np
import psutil
import utils
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import concatenate_lazy_data
from aind_large_scale_prediction.io import ImageReaderFactory
from neuroglancer import CoordinateSpace
from torch import squeeze

from ._shared.types import ArrayLike, PathLike
# from lazy_deskewing import (create_dispim_config, create_dispim_transform, lazy_deskewing)
from .traditional_detection.puncta_detection_optimized import (
    prune_blobs, traditional_3D_spot_detection)
from .utils.generate_precomputed_format import generate_precomputed_spots


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

    mask[mask > 0] = 1
    mask = mask.astype(np.uint8)

    data = data * mask

    return data.astype(orig_dtype)


def execute_worker(
    data: ArrayLike,
    batch_super_chunk: Tuple[slice],
    batch_internal_slice: Tuple[slice],
    spot_parameters: Dict,
    logger: logging.Logger,
) -> np.array:
    """
    Function that executes each worker. It takes
    the combined gradients and follows the flows.

    Parameters
    ----------
    data: ArrayLike
        Data to process.

    batch_super_chunk: Tuple[slice]
        Slices of the super chunk loaded in shared memory.

    batch_internal_slice: Tuple[slice]
        Internal slice of the current chunk of data. This
        is a local coordinate system based on the super chunk.

    spot_parameters: Dict
        Spot detection parameters.

    logger: logging.Logger
        Logging object
    """
    curr_pid = os.getpid()
    data = squeeze(data)

    if data.shape[-4] == 2:
        data = apply_mask(data=data[:, 0, ...], mask=data[:, 1, ...])

    data_block_cupy = cupy.asarray(data)
    worker_spots = None

    for batch_idx in range(0, data.shape[0]):
        logger.info(
            f"Worker [{curr_pid}] Processing inner batch {batch_idx} out of {data.shape[0]}"
        )

        # Making sure CuPy it's running in the correct device
        spots = traditional_3D_spot_detection(
            data_block=data_block_cupy[batch_idx, ...],
            background_percentage=spot_parameters["background_percentage"],
            sigma_zyx=spot_parameters["sigma_zyx"],
            pad_size=spot_parameters["pad_size"],
            min_zyx=spot_parameters["min_zyx"],
            filt_thresh=spot_parameters["filt_thresh"],
            raw_thresh=spot_parameters["raw_thresh"],
            context_radius=spot_parameters["context_radius"],
            radius_confidence=spot_parameters["radius_confidence"],
            logger=logger,
        )

        curr_spots = None
        if spots is None:
            logger.info(
                f"Worker [{curr_pid}] - No spots found in inner batch {batch_idx}"
            )

        else:
            logger.info(
                f"Worker [{curr_pid}] - Found {len(spots)} spots for in inner batch {batch_idx}"
            )

            # Getting current spots
            curr_spots = recover_global_position(
                super_chunk_slice=batch_super_chunk,
                internal_slice=batch_internal_slice,
                zyx_points=spots,
            )

            # Adding spots to the worker batch
            if worker_spots is None:
                worker_spots = curr_spots.copy()

            else:
                worker_spots = np.append(
                    worker_spots,
                    curr_spots,
                    axis=0,
                )

    return worker_spots


def _execute_worker(params):
    """
    Worker interface to provide parameters
    """
    execute_worker(**params)


def z1_puncta_detection(
    dataset_path: PathLike,
    multiscale: str,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    batch_size: 96,
    output_folder: str,
    spot_parameters: Dict,
    logger: logging.Logger,
    super_chunksize: Optional[Tuple[int, ...]] = None,
    segmentation_mask_path: Optional[PathLike] = None,
):
    """
    Chunked puncta detection

    Parameters
    ----------
    dataset_path: PathLike
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

    segmentation_mask_path: Optional[PathLike]
        Path where the segmentation mask is stored. It could
        be a local path or in a S3 path.
        Default None
    """
    co_cpus = int(utils.get_code_ocean_cpu_limit())

    if n_workers > co_cpus:
        raise ValueError(f"Provided workers {n_workers} > current workers {co_cpus}")

    logger.info(f"{20*'='} Running puncta detection {20*'='}")
    logger.info(f"Output folder: {output_folder}")

    utils.print_system_information(logger)

    logger.info(f"Processing dataset {dataset_path} with mulsticale {multiscale}")

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

    logger.info("Creating chunked data loader")
    shm_memory = psutil.virtual_memory()
    logger.info(f"Shared memory information: {shm_memory}")

    device = None

    pin_memory = True
    if device is not None:
        pin_memory = False
        multiprocessing.set_start_method("spawn", force=True)

    if segmentation_mask_path:
        logger.info(f"Using segmentation mask in {segmentation_mask_path}")
        lazy_data = concatenate_lazy_data(
            dataset_paths=[dataset_path, segmentation_mask_path],
            multiscale=multiscale,
            concat_axis=-4,
        )

    else:
        lazy_data = (
            ImageReaderFactory()
            .create(data_path=dataset_path, parse_path=False, multiscale=multiscale)
            .as_dask_array()
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

    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(f"Number of batches: {total_batches}")
    spots_global_coordinate = None

    # Setting exec workers to CO CPUs
    exec_n_workers = co_cpus

    # Create a pool of processes
    pool = multiprocessing.Pool(processes=exec_n_workers)

    # Variables for multiprocessing
    picked_blocks = []
    curr_picked_blocks = 0

    logger.info(f"Number of workers processing data: {exec_n_workers}")

    with cupy.cuda.Device(device=device):
        with cupy.cuda.Stream.null:
            for i, sample in enumerate(zarr_data_loader):
                logger.info(
                    f"Batch {i}: {sample.batch_tensor.shape} - Pinned?: {sample.batch_tensor.is_pinned()} - dtype: {sample.batch_tensor.dtype} - device: {sample.batch_tensor.device}"
                )

                start_spot_time = time()

                picked_blocks.append(
                    {
                        "data": sample.batch_tensor,
                        "batch_super_chunk": sample.batch_super_chunk[0],
                        "batch_internal_slice": sample.batch_internal_slice,
                        "spot_parameters": spot_parameters,
                        "logger": logger,
                    }
                )
                curr_picked_blocks += 1

                if curr_picked_blocks == exec_n_workers:
                    # Assigning blocks to execution workers
                    jobs = [
                        pool.apply_async(_execute_worker, args=(picked_block,))
                        for picked_block in picked_blocks
                    ]

                    logger.info(
                        f"Dispatcher PID {os.getpid()} dispatching {len(jobs)} jobs"
                    )

                    # Wait for all processes to finish
                    worker_spots = [job.get() for job in jobs]
                    worker_spots = [w for w in worker_spots if w is not None]

                    # Concatenate worker spots
                    worker_spots = np.concatenate(worker_spots, axis=0)

                    # Setting variables back to init
                    curr_picked_blocks = 0
                    picked_blocks = []

                    # Adding picked spots to global list of spots
                    if spots_global_coordinate is None:
                        spots_global_coordinate = worker_spots.copy()

                    else:
                        spots_global_coordinate = np.append(
                            spots_global_coordinate,
                            worker_spots,
                            axis=0,
                        )

                end_spot_time = time()
                logger.info(
                    f"Time processing batch {i}: {end_spot_time - start_spot_time}"
                )

                if i + samples_per_iter > total_batches:
                    logger.info(
                        f"Not enough samples to retrieve from workers, remaining: {i + samples_per_iter - total_batches}"
                    )
                    break

    if curr_picked_blocks != 0:
        logger.info(f"Blocks not processed inside of loop: {curr_picked_blocks}")
        # Assigning blocks to execution workers
        jobs = [
            pool.apply_async(_execute_worker, args=(picked_block,))
            for picked_block in picked_blocks
        ]

        logger.info(f"Dispatcher PID {os.getpid()} dispatching {len(jobs)} jobs")

        # Wait for all processes to finish
        worker_spots = [job.get() for job in jobs]
        worker_spots = [w for w in worker_spots if w is not None]

        # Concatenate worker spots
        worker_spots = np.concatenate(worker_spots, axis=0)

        # Setting variables back to init
        curr_picked_blocks = 0
        picked_blocks = []

        # Adding picked spots to global list of spots
        if spots_global_coordinate is None:
            spots_global_coordinate = worker_spots.copy()

        else:
            spots_global_coordinate = np.append(
                spots_global_coordinate,
                worker_spots,
                axis=0,
            )

    end_time = time()

    # Final prunning, might be spots in boundaries where spots where splitted
    start_final_prunning_time = time()
    spots_global_coordinate_prunned = prune_blobs(
        blobs_array=spots_global_coordinate,
        distance=spot_parameters["min_zyx"][-1] + spot_parameters["radius_confidence"],
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
