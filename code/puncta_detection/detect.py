"""
Large-scale puncta detection using single GPU
"""

import logging
import multiprocessing
import os
# from functools import partial
from time import time
from typing import Dict, Optional, Tuple

import cupy
import numpy as np
import psutil
import torch
from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    concatenate_lazy_data, recover_global_position, unpad_global_coords)
from aind_large_scale_prediction.io import ImageReaderFactory
from neuroglancer import CoordinateSpace

from ._shared.types import ArrayLike, PathLike
from .traditional_detection.puncta_detection_optimized import (
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

            curr_spots = spots.copy().astype(np.int32)
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


def producer(
    producer_queue,
    zarr_data_loader,
    logger,
    n_consumers,
):
    """
    Function that sends blocks of data to
    the queue to be acquired by the workers.

    Parameters
    ----------
    producer_queue: multiprocessing.Queue
        Multiprocessing queue where blocks
        are sent to be acquired by workers.

    zarr_data_loader: DataLoader
        Zarr data loader

    logger: logging.Logger
        Logging object

    n_consumers: int
        Number of consumers
    """
    # total_samples = sum(zarr_dataset.internal_slice_sum)
    worker_pid = os.getpid()

    logger.info(f"Starting producer queue: {worker_pid}")
    for i, sample in enumerate(zarr_data_loader):

        producer_queue.put(
            {
                "data_block": sample.batch_tensor.numpy()[0, ...],
                "i": i,
                "batch_super_chunk": sample.batch_super_chunk[0],
                "batch_internal_slice": sample.batch_internal_slice,
            },
            block=True,
        )
        logger.info(f"[+] Worker {worker_pid} setting block {i}")

    for i in range(n_consumers):
        producer_queue.put(None, block=True)

    # zarr_dataset.lazy_data.shape
    logger.info(f"[+] Worker {worker_pid} -> Producer finished producing data.")


def consumer(
    queue,
    zarr_dataset,
    worker_params,
    results_dict,
):
    """
    Function executed in every worker
    to acquire data.

    Parameters
    ----------
    queue: multiprocessing.Queue
        Multiprocessing queue where blocks
        are sent to be acquired by workers.

    zarr_dataset: ArrayLike
        Zarr dataset

    worker_params: dict
        Worker parametes to execute a function.

    results_dict: multiprocessing.Dict
        Results dictionary where outputs
        are stored.
    """
    worker_spots = None
    logger = worker_params["logger"]
    worker_pid = os.getpid()
    logger.info(f"Starting consumer worker -> {worker_pid}")

    # Start processing
    total_samples = sum(zarr_dataset.internal_slice_sum)

    # Getting data until queue is empty
    while True:
        streamed_dict = queue.get(block=True)

        if streamed_dict is None:
            logger.info(f"[-] Worker {worker_pid} -> Turn off signal received...")
            break

        logger.info(
            f"[-] Worker {worker_pid} -> Consuming {streamed_dict['i']} - {streamed_dict['data_block'].shape} - Super chunk val: {zarr_dataset.curr_super_chunk_pos.value} - internal slice sum: {total_samples}"
        )

        # Getting spots
        worker_response = execute_worker(
            data=streamed_dict["data_block"],
            batch_super_chunk=streamed_dict["batch_super_chunk"],
            batch_internal_slice=streamed_dict["batch_internal_slice"],
            spot_parameters=worker_params["spot_parameters"],
            overlap_prediction_chunksize=worker_params["overlap_prediction_chunksize"],
            dataset_shape=worker_params["dataset_shape"],
            logger=worker_params["logger"],
        )

        if worker_response is not None:

            if worker_spots is None:
                worker_spots = worker_response.copy()

            else:
                worker_spots = np.append(
                    worker_spots,
                    worker_response,
                    axis=0,
                )

    logger.info(f"[-] Worker {worker_pid} -> Consumer finished consuming data.")
    results_dict[worker_pid] = worker_spots


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

    logger.info(
        f"Running puncta detection in chunked data. Prediction chunksize: {prediction_chunksize} - Overlap chunksize: {overlap_prediction_chunksize}"
    )

    start_time = time()

    total_batches = sum(zarr_dataset.internal_slice_sum) / batch_size

    logger.info(f"Number of batches: {total_batches}")
    spots_global_coordinate = None

    # Setting exec workers to CO CPUs
    exec_n_workers = co_cpus

    # Create consumer processes
    factor = 10

    # Create a multiprocessing queue
    producer_queue = multiprocessing.Queue(maxsize=exec_n_workers * factor)
    results_dict = manager.dict()

    worker_params = {
        "overlap_prediction_chunksize": overlap_prediction_chunksize,
        "dataset_shape": zarr_dataset.lazy_data.shape,
        "spot_parameters": spot_parameters,
        "logger": logger,
    }

    logger.info(f"Setting up {exec_n_workers} workers...")
    consumers = [
        multiprocessing.Process(
            target=consumer,
            args=(
                producer_queue,
                zarr_dataset,
                worker_params,
                results_dict,
            ),
        )
        for _ in range(exec_n_workers)
    ]

    # Start consumer processes
    for consumer_process in consumers:
        consumer_process.start()

    # Main process acts as the producer
    producer(producer_queue, zarr_data_loader, logger, exec_n_workers)

    # Wait for consumer processes to finish
    for consumer_process in consumers:
        consumer_process.join()

    # All spots must be in the shared dictionary
    spots_global_coordinate = np.concatenate(list(results_dict.values()), axis=0)

    end_time = time()

    if spots_global_coordinate is None:
        logger.info("No spots found!")

    else:
        spots_global_coordinate = spots_global_coordinate.astype(np.int32)
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
            spots=spots_global_coordinate_prunned[:, :3],  # Only ZYX locations
            path=f"{output_folder}/precomputed",
            res=coord_space,
        )

        logger.info(f"Processing time: {end_time - start_time} seconds")

        # Saving spots
        np.save(f"{output_folder}/spots.npy", spots_global_coordinate_prunned)

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
