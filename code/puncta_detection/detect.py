"""
Large-scale puncta detection using single GPU
"""

import gc
import logging
import multiprocessing
import os
# from functools import partial
from time import time
from typing import Dict, List, Optional, Tuple

import cupy
import numpy as np
import pandas as pd
import psutil
import torch
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    concatenate_lazy_data, recover_global_position, unpad_global_coords)
from aind_large_scale_prediction.io import ImageReaderFactory
from neuroglancer import CoordinateSpace

from ._shared.types import ArrayLike, PathLike
# from lazy_deskewing import (create_dispim_config, create_dispim_transform, lazy_deskewing)
from .traditional_detection.puncta_detection import (
    prune_blobs, traditional_3D_spot_detection)
from .utils import utils
from .utils.generate_precomputed_format import generate_precomputed_spots


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
    if isinstance(mask, torch.Tensor):
        mask = mask.to(torch.uint8)

    else:
        mask = mask.astype(np.uint8)

    data = data * mask

    if isinstance(data, torch.Tensor):
        data = data.to(orig_dtype)

    else:
        data = data.astype(orig_dtype)

    return data


def remove_points_in_pad_area(
    points: ArrayLike, unpadded_slices: Tuple[slice]
) -> ArrayLike:
    """
    Removes points in padding area. The padding is provided
    by the scheduler as well as the unpadded slices which
    will be used to remove points in those areas.

    Parameters
    ----------
    points: ArrayLike
        3D points in the chunk of data. When masks are provided,
        points will be 4D with an extra dimension for the mask id
        which is not modified.

    unpadded_slices: Tuple[slice]
        Slices that point to the non-overlapping area between chunks
        of data.

    Returns
    -------
    ArrayLike
        Points within the non-overlapping area.
    """

    # Validating seeds are within block boundaries
    unpadded_points = points[
        (points[:, 0] >= unpadded_slices[0].start)  # within Z boundaries
        & (points[:, 0] <= unpadded_slices[0].stop)
        & (points[:, 1] >= unpadded_slices[1].start)  # Within Y boundaries
        & (points[:, 1] <= unpadded_slices[1].stop)
        & (points[:, 2] >= unpadded_slices[2].start)  # Within X boundaries
        & (points[:, 2] <= unpadded_slices[2].stop)
    ]

    return unpadded_points


def execute_worker(
    data: ArrayLike,
    batch_super_chunk: Tuple[slice],
    batch_internal_slice: Tuple[slice],
    spot_parameters: Dict,
    overlap_prediction_chunksize: Tuple[int],
    dataset_shape: Tuple[int],
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

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Array with the global location of the identified points.
    """
    curr_pid = os.getpid()
    mask = None
    # (Batch, channels, Z, Y, X)
    if len(data.shape) == 5 and data.shape[-4] == 2:
        mask = data[:, 1, ...]
        data = data[:, 0, ...]
        data = apply_mask(data=data, mask=mask.detach().clone())

    data_block_cupy = cupy.asarray(data)
    global_worker_spots = None

    # Processing batch
    for batch_idx in range(0, data.shape[0]):
        curr_block = cupy.squeeze(data_block_cupy[batch_idx, ...])
        logger.info(
            f"Worker [{curr_pid}] Processing inner batch {batch_idx} out of {data.shape[0]} - Data shape: {data.shape} - Current block: {curr_block.shape}"
        )

        # Making sure CuPy it's running in the correct device
        spots = traditional_3D_spot_detection(
            data_block=curr_block,
            background_percentage=spot_parameters["background_percentage"],
            sigma_zyx=spot_parameters["sigma_zyx"],
            min_zyx=spot_parameters["min_zyx"],
            filt_thresh=spot_parameters["filt_thresh"],
            raw_thresh=spot_parameters["raw_thresh"],
            context_radius=spot_parameters["context_radius"],
            radius_confidence=spot_parameters["radius_confidence"],
            logger=logger,
        )

        # Adding spots to current batch list
        curr_spots = None
        if spots is None:
            logger.info(
                f"Worker [{curr_pid}] - No spots found in inner batch {batch_idx}"
            )

        else:

            # Recover global position of internal chunk
            (
                global_coord_pos,
                global_coord_positions_start,
                global_coord_positions_end,
            ) = recover_global_position(
                super_chunk_slice=batch_super_chunk,
                internal_slices=batch_internal_slice,
            )

            unpadded_global_slice, unpadded_local_slice = unpad_global_coords(
                global_coord_pos=global_coord_pos[-3:],
                block_shape=curr_block.shape[-3:],
                overlap_prediction_chunksize=overlap_prediction_chunksize[-3:],
                dataset_shape=dataset_shape[-3:],  # zarr_dataset.lazy_data.shape,
            )

            if mask is not None:
                # Getting spots IDs, adding mask ID to the spot as extra value at the end
                mask = torch.squeeze(mask)
                mask_ids = np.expand_dims(
                    mask[spots[:, 0], spots[:, 1], spots[:, 2]], axis=0
                )
                spots = np.append(spots.T, mask_ids, axis=0).T

            curr_spots = spots.copy().astype(np.float32)
            # Converting to global coordinates, only to ZYX position, leaving mask ID if exists
            curr_spots[:, :3] = np.array(global_coord_positions_start)[
                :, -3:
            ] + np.array(spots[:, :3])

            # Removing points within pad area
            curr_spots = remove_points_in_pad_area(
                points=curr_spots, unpadded_slices=unpadded_global_slice
            )

            logger.info(
                f"Worker {curr_pid}: Found {len(curr_spots)} spots for in inner batch {batch_idx} - Internal pos: {batch_internal_slice} - Global coords: {global_coord_pos} - upadded global coords: {unpadded_global_slice}"
            )

            # Adding spots to the worker batch
            if global_worker_spots is None:
                global_worker_spots = curr_spots.copy()

            else:
                global_worker_spots = np.append(
                    global_worker_spots,
                    curr_spots,
                    axis=0,
                )

    return global_worker_spots


def _execute_worker(params: Dict):
    """
    Worker interface to provide parameters

    Parameters
    ----------
    params: Dict
        Dictionary with the parameters to provide
        to the execution function.
    """
    return execute_worker(**params)


def helper_submit_jobs(
    picked_blocks: List[dict], pool: multiprocessing.Pool
) -> ArrayLike:
    """
    Helper function that submits jobs and collects
    the results from the jobs.

    Parameters
    ----------
    picked_blocks: List[dict]
        List with the parameters that will be passed
        to the execute_worker function.

    pool: multiprocessing.Pool
        Pool of workers

    Returns
    -------
    Returns the concatenated spots from workers
    """
    # Assigning blocks to execution workers
    jobs = [
        pool.apply_async(_execute_worker, args=(picked_block,))
        for picked_block in picked_blocks
    ]

    global_workers_spots = []

    # Wait for all processes to finish
    for job in jobs:
        worker_response = job.get()

        if worker_response is not None:
            # Global coordinate points
            global_workers_spots.append(worker_response)

    # Concatenate worker spots
    if len(global_workers_spots):
        # Concatenating lists to numpy array
        global_workers_spots = np.concatenate(global_workers_spots, axis=0)
    else:
        global_workers_spots = np.array(global_workers_spots)

    return global_workers_spots


def z1_puncta_detection(
    dataset_path: PathLike,
    multiscale: str,
    prediction_chunksize: Tuple[int, ...],
    target_size_mb: int,
    n_workers: int,
    axis_pad: int,
    batch_size: int,
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

    overlap_prediction_chunksize = (axis_pad, axis_pad, axis_pad)
    if segmentation_mask_path:
        logger.info(f"Using segmentation mask in {segmentation_mask_path}")
        lazy_data = concatenate_lazy_data(
            dataset_paths=[dataset_path, segmentation_mask_path],
            multiscales=[multiscale, "0"],
            concat_axis=-4,
        )
        overlap_prediction_chunksize = (0, axis_pad, axis_pad, axis_pad)
        prediction_chunksize = (lazy_data.shape[-4],) + prediction_chunksize

        logger.info(
            f"Segmentation mask provided! New prediction chunksize: {prediction_chunksize} - New overlap: {overlap_prediction_chunksize}"
        )

    else:
        # No segmentation mask
        lazy_data = (
            ImageReaderFactory()
            .create(data_path=dataset_path, parse_path=False, multiscale=multiscale)
            .as_dask_array()
        )

    image_metadata = (
        ImageReaderFactory()
        .create(data_path=dataset_path, parse_path=False, multiscale=multiscale)
        .metadata()
    )

    logger.info(f"Full image metadata: {image_metadata}")

    image_metadata = utils.parse_zarr_metadata(
        metadata=image_metadata, multiscale=multiscale
    )

    logger.info(f"Filtered Image metadata: {image_metadata}")

    zarr_data_loader, zarr_dataset = create_data_loader(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        prediction_chunksize=prediction_chunksize,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
        n_workers=1,
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

    logger.info(
        f"Running puncta detection in chunked data. Prediction chunksize: {prediction_chunksize} - Overlap chunksize: {overlap_prediction_chunksize}"
    )

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

    with multiprocessing.Pool(processes=exec_n_workers) as pool:
        # pool = multiprocessing.Pool(processes=exec_n_workers)

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

                    # start_spot_time = time()

                    picked_blocks.append(
                        {
                            "data": sample.batch_tensor,
                            "batch_super_chunk": sample.batch_super_chunk[0],
                            "batch_internal_slice": sample.batch_internal_slice,
                            "overlap_prediction_chunksize": overlap_prediction_chunksize,
                            "dataset_shape": zarr_dataset.lazy_data.shape,
                            "spot_parameters": spot_parameters,
                            "logger": logger,
                        }
                    )
                    curr_picked_blocks += 1

                    if curr_picked_blocks == exec_n_workers:
                        logger.info(
                            f"Dispatcher PID {os.getpid()} dispatching {curr_picked_blocks} jobs"
                        )
                        global_workers_spots = helper_submit_jobs(picked_blocks, pool)

                        # Adding picked spots to global list of spots
                        # I could've done this inside helper_submit_jobs but
                        # it'd be a bad programming practice
                        if (
                            spots_global_coordinate is None
                            and global_workers_spots.shape[0]
                        ):
                            spots_global_coordinate = global_workers_spots.copy()

                        elif global_workers_spots.shape[0]:
                            print(
                                global_workers_spots.shape, global_workers_spots.shape
                            )
                            spots_global_coordinate = np.append(
                                spots_global_coordinate,
                                global_workers_spots,
                                axis=0,
                            )

                        # Setting variables back to init
                        curr_picked_blocks = 0
                        picked_blocks = []

                    # end_spot_time = time()
                    # logger.info(
                    #     f"Time processing batch {i}: {end_spot_time - start_spot_time}"
                    # )

                    if i + samples_per_iter > total_batches:
                        logger.info(
                            f"Not enough samples to retrieve from workers, remaining: {i + samples_per_iter - total_batches}"
                        )
                        break

        if curr_picked_blocks != 0:
            logger.info(f"Blocks not processed inside of loop: {curr_picked_blocks}")
            logger.info(
                f"Dispatcher PID {os.getpid()} dispatching {curr_picked_blocks} jobs"
            )
            global_workers_spots = helper_submit_jobs(picked_blocks, pool)
            # Adding picked spots to global list of spots
            # I could've done this inside helper_submit_jobs but
            # it'd be a bad programming practice
            if spots_global_coordinate is None and global_workers_spots.shape[0]:
                spots_global_coordinate = global_workers_spots.copy()

            elif global_workers_spots.shape[0]:
                print(global_workers_spots.shape, global_workers_spots.shape)
                spots_global_coordinate = np.append(
                    spots_global_coordinate,
                    global_workers_spots,
                    axis=0,
                )

            # Setting variables back to init
            curr_picked_blocks = 0
            picked_blocks = []

    end_time = time()

    if spots_global_coordinate is None:
        logger.info("No spots found!")

    else:
        spots_global_coordinate = spots_global_coordinate.astype(np.float32)
        # Final prunning, might be spots in boundaries where spots where splitted
        start_final_prunning_time = time()
        spots_global_coordinate_prunned, removed_pos = prune_blobs(
            blobs_array=spots_global_coordinate.copy(),  # Prunning only ZYX locations, careful with Masks IDs
            distance=spot_parameters["min_zyx"][-1]
            + spot_parameters["radius_confidence"],
        )
        end_final_prunning_time = time()

        logger.info(
            f"Time taken for final prunning {end_final_prunning_time - start_final_prunning_time} before: {len(spots_global_coordinate)} After: {len(spots_global_coordinate_prunned)}"
        )

        # TODO add chunked precomputed format for points with multiscales
        coord_space = CoordinateSpace(
            names=["z", "y", "x"],
            units=["um", "um", "um"],
            scales=[
                image_metadata["axes"]["z"]["scale"],
                image_metadata["axes"]["y"]["scale"],
                image_metadata["axes"]["x"]["scale"],
            ],
        )

        logger.info(f"Neuroglancer coordinate space: {coord_space}")
        generate_precomputed_spots(
            spots=spots_global_coordinate_prunned[:, :3].copy(),  # Only ZYX locations
            path=f"{output_folder}/precomputed",
            res=coord_space,
        )

        logger.info(f"Processing time: {end_time - start_time} seconds")

        # Saving spots
        np.save(f"{output_folder}/spots.npy", spots_global_coordinate_prunned)

        columns = ["Z", "Y", "X", "Z_center", "Y_center", "X_center", "dist", "r"]
        sort_column = "Z"
        int_columns = ["Z", "Y", "X"]

        # If segmentation mask is provided, let's sort by ID
        if segmentation_mask_path:
            columns.append("SEG_ID")
            sort_column = "SEG_ID"
            int_columns.append("SEG_ID")

        spots_df = pd.DataFrame(spots_global_coordinate_prunned, columns=columns)

        spots_df[int_columns] = spots_df[int_columns].astype("int")
        spots_df = spots_df.sort_values(by=sort_column)

        # Saving spots
        spots_df.to_csv(
            f"{output_folder}/spots.csv",
            index=False,
        )

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            output_folder,
            "puncta_detection",
        )
