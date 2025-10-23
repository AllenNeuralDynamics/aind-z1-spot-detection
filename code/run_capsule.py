"""
Scripts that runs the Code Ocean capsule - MULTI-TILE VERSION

This version handles both single-tile and multi-tile processing.
The dispatcher creates a JSON manifest that specifies which mode to use.
"""

import os
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from puncta_detection.detect import z1_puncta_detection
from puncta_detection.utils import utils


# ============================================================================
# UNCHANGED: Original helper function
# ============================================================================

def strip_all_suffixes(path: Path) -> str:
    """
    Strips all suffixes from path

    Parameters
    ----------
    path: Path
        Dataset path

    Returns
    -------
    str
        String with all stripped suffixes
    """
    name = path.name
    for suffix in path.suffixes:
        name = name[: -len(suffix)]
    return name


# ============================================================================
# NEW: Helper functions for multi-tile processing
# ============================================================================

def get_tile_dimensions(tile_path: Path, multiscale: str = "0") -> Tuple[int, int, int]:
    """
    Get the dimensions of a tile from zarr metadata.
    
    Parameters
    ----------
    tile_path : Path
        Path to the tile zarr file
    multiscale : str
        Multiscale level to read
        
    Returns
    -------
    Tuple[int, int, int]
        Tile dimensions in (Z, Y, X)
    """
    from aind_large_scale_prediction.io import ImageReaderFactory
    
    try:
        reader = ImageReaderFactory().create(
            data_path=str(tile_path), 
            parse_path=False, 
            multiscale=multiscale
        )
        metadata = reader.metadata()
        
        # Try to get shape from metadata
        if 'shape' in metadata:
            shape = metadata['shape']
        elif 'axes' in metadata:
            # Alternative: calculate from axes info
            shape = tuple(metadata['axes'][ax]['length'] for ax in ['z', 'y', 'x'])
        else:
            # Fallback: read from zarr array directly
            import zarr
            zarray = zarr.open(str(tile_path), mode='r')
            if multiscale != "0":
                zarray = zarray[int(multiscale)]
            shape = zarray.shape
        
        # Return Z, Y, X (last 3 dimensions)
        return tuple(shape[-3:])
        
    except Exception as e:
        raise ValueError(f"Could not determine tile dimensions for {tile_path}: {e}")


def transform_spots_to_global_coordinates(
    spots_df: pd.DataFrame,
    tile_indices: Optional[Tuple[int, int, int]],
    tile_dimensions: Tuple[int, int, int],
    tile_name: str,
) -> pd.DataFrame:
    """
    Transform spot coordinates from tile-local to global coordinate system.
    
    Parameters
    ----------
    spots_df : pd.DataFrame
        DataFrame with spot detections in local tile coordinates
    tile_indices : Optional[Tuple[int, int, int]]
        Tile index offset (z_idx, y_idx, x_idx). None if not available.
    tile_dimensions : Tuple[int, int, int]
        Physical dimensions of tile in voxels (z, y, x)
    tile_name : str
        Name of the tile for traceability
        
    Returns
    -------
    pd.DataFrame
        DataFrame with spots in global coordinates (if tile_indices provided)
        or original coordinates with tile_name added
    """
    spots_global = spots_df.copy()
    
    # Add tile identifier for traceability
    spots_global['tile_name'] = tile_name
    
    if tile_indices is not None:
        # Calculate global offset in voxels
        z_offset = tile_indices[0] * tile_dimensions[0]
        y_offset = tile_indices[1] * tile_dimensions[1]
        x_offset = tile_indices[2] * tile_dimensions[2]
        
        # Transform coordinates
        spots_global['Z'] = spots_global['Z'] + z_offset
        spots_global['Y'] = spots_global['Y'] + y_offset
        spots_global['X'] = spots_global['X'] + x_offset
        
        # Transform center coordinates if they exist
        if 'Z_center' in spots_global.columns:
            spots_global['Z_center'] = spots_global['Z_center'] + z_offset
            spots_global['Y_center'] = spots_global['Y_center'] + y_offset
            spots_global['X_center'] = spots_global['X_center'] + x_offset
        
        # Add detailed tile position info
        spots_global['tile_z_idx'] = tile_indices[0]
        spots_global['tile_y_idx'] = tile_indices[1]
        spots_global['tile_x_idx'] = tile_indices[2]
        
        spots_global['tile_z_offset'] = z_offset
        spots_global['tile_y_offset'] = y_offset
        spots_global['tile_x_offset'] = x_offset
    
    return spots_global


def process_single_tile(
    data_path: Path,
    tile_info: Dict,
    tile_idx: int,
    base_puncta_params: Dict,
    main_output_folder: Path,
    main_logger,
    multiscale: str,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """
    Process a single tile for spot detection.
    
    Parameters
    ----------
    data_path : Path
        Path to the tile zarr file
    tile_info : Dict
        Tile information from manifest
    tile_idx : int
        Index of this tile in processing order
    base_puncta_params : Dict
        Base spot detection parameters
    main_output_folder : Path
        Main output folder
    main_logger : logging.Logger
        Logger instance
    multiscale : str
        Multiscale level
        
    Returns
    -------
    Tuple[Optional[pd.DataFrame], Dict]
        Transformed spots dataframe (or None) and processing metadata
    """
    tile_name = tile_info['tile_name']
    
    main_logger.info(f"\n{'='*80}")
    main_logger.info(f"Processing tile {tile_idx + 1}: {tile_name}")
    main_logger.info(f"{'='*80}\n")
    
    # Create tile-specific output folder
    tile_output_folder = main_output_folder.joinpath(
        f"tile_{tile_idx:03d}_{tile_name}"
    )
    utils.create_folder(dest_dir=str(tile_output_folder), verbose=True)
    
    # Create tile-specific logger
    tile_logger = utils.create_logger(
        output_log_path=str(tile_output_folder)
    )
    
    # Get tile dimensions for coordinate transformation
    try:
        tile_dimensions = get_tile_dimensions(data_path, multiscale=multiscale)
        main_logger.info(f"Tile dimensions (Z,Y,X): {tile_dimensions}")
    except Exception as e:
        main_logger.error(f"Could not determine tile dimensions: {e}")
        tile_dimensions = (0, 0, 0)
    
    # Prepare metadata
    tile_metadata = {
        'tile_idx': tile_idx,
        'tile_name': tile_name,
        'tile_path': str(data_path),
        'tile_indices': tile_info.get('tile_indices'),
        'tile_dimensions': tile_dimensions,
        'has_position_info': tile_info.get('has_position_info', False),
    }
    
    # Prepare parameters for this tile
    tile_puncta_params = base_puncta_params.copy()
    tile_puncta_params['dataset_path'] = str(data_path)
    tile_puncta_params['output_folder'] = tile_output_folder
    tile_puncta_params['logger'] = tile_logger
    
    try:
        # Run spot detection
        main_logger.info(f"Starting spot detection for tile {tile_idx}")
        z1_puncta_detection(**tile_puncta_params)
        main_logger.info(f"Completed spot detection for tile {tile_idx}")
        
        # Load the generated spots CSV
        tile_spots_csv = tile_output_folder / "spots.csv"
        
        if tile_spots_csv.exists():
            spots_df = pd.read_csv(tile_spots_csv)
            n_spots = len(spots_df)
            main_logger.info(f"Tile {tile_idx}: Found {n_spots} spots")
            
            # Transform to global coordinates if position info available
            if tile_info.get('has_position_info') and tile_info.get('tile_indices'):
                main_logger.info(
                    f"Transforming coordinates using tile indices: {tile_info['tile_indices']}"
                )
                spots_df_global = transform_spots_to_global_coordinates(
                    spots_df=spots_df,
                    tile_indices=tile_info['tile_indices'],
                    tile_dimensions=tile_dimensions,
                    tile_name=tile_name,
                )
            else:
                main_logger.warning(
                    f"No position information for tile {tile_idx}. "
                    "Using local coordinates only."
                )
                spots_df_global = spots_df.copy()
                spots_df_global['tile_name'] = tile_name
            
            # Save the transformed coordinates back to the tile folder
            transformed_csv = tile_output_folder / "spots_global_coords.csv"
            spots_df_global.to_csv(transformed_csv, index=False)
            main_logger.info(f"Saved global coordinates to {transformed_csv}")
            
            tile_metadata['n_spots'] = n_spots
            return spots_df_global, tile_metadata
            
        else:
            main_logger.warning(f"No spots.csv found for tile {tile_idx}")
            tile_metadata['n_spots'] = 0
            return None, tile_metadata
            
    except Exception as e:
        main_logger.error(
            f"Error processing tile {tile_idx}: {str(e)}", 
            exc_info=True
        )
        tile_metadata['error'] = str(e)
        tile_metadata['n_spots'] = 0
        return None, tile_metadata


def process_multi_tile_detection(
    tiles_info: List[Dict],
    base_puncta_params: Dict,
    output_folder: Path,
    main_logger,
    multiscale: str,
) -> Tuple[List[pd.DataFrame], List[Dict]]:
    """
    Process spot detection on multiple tiles.
    
    Parameters
    ----------
    tiles_info : List[Dict]
        List of tile information dictionaries from the manifest
    base_puncta_params : Dict
        Base spot detection parameters
    output_folder : Path
        Main output folder for results
    main_logger : logging.Logger
        Logger instance
    multiscale : str
        Multiscale level to process
        
    Returns
    -------
    Tuple[List[pd.DataFrame], List[Dict]]
        List of spot dataframes (one per tile) and list of processing metadata
    """
    all_spots_dfs = []
    tile_metadata = []
    
    for tile_idx, tile_info in enumerate(tiles_info):
        data_path = Path(tile_info['path'])
        
        spots_df, metadata = process_single_tile(
            data_path=data_path,
            tile_info=tile_info,
            tile_idx=tile_idx,
            base_puncta_params=base_puncta_params,
            main_output_folder=output_folder,
            main_logger=main_logger,
            multiscale=multiscale,
        )
        
        if spots_df is not None:
            all_spots_dfs.append(spots_df)
        
        tile_metadata.append(metadata)
    
    return all_spots_dfs, tile_metadata


def concatenate_and_save_results(
    all_spots_dfs: List[pd.DataFrame],
    tile_metadata: List[Dict],
    output_folder: Path,
    channel_name: str,
    logger,
):
    """
    Concatenate spot detection results and save combined outputs.
    
    Parameters
    ----------
    all_spots_dfs : List[pd.DataFrame]
        List of spot dataframes from all tiles
    tile_metadata : List[Dict]
        Metadata about each processed tile
    output_folder : Path
        Output folder for combined results
    channel_name : str
        Channel name for output file naming
    logger : logging.Logger
        Logger instance
    """
    if not all_spots_dfs:
        logger.warning("No spots detected in any tiles!")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info("Concatenating results from all tiles")
    logger.info(f"{'='*80}\n")
    
    # Concatenate all spots
    combined_spots_df = pd.concat(all_spots_dfs, ignore_index=True)
    
    # Sort by Z coordinate (global if available)
    combined_spots_df = combined_spots_df.sort_values(by='Z')
    
    logger.info(f"Total spots across all tiles: {len(combined_spots_df)}")
    
    # Save combined results with channel in filename
    combined_csv_path = output_folder / f"ch_{channel_name}_all_tiles_spots.csv"
    combined_spots_df.to_csv(combined_csv_path, index=False)
    logger.info(f"Saved combined spots to {combined_csv_path}")
    
    # Save tile metadata
    tile_metadata_df = pd.DataFrame(tile_metadata)
    tile_metadata_path = output_folder / f"ch_{channel_name}_tile_metadata.csv"
    tile_metadata_df.to_csv(tile_metadata_path, index=False)
    logger.info(f"Saved tile metadata to {tile_metadata_path}")
    
    # Generate and save summary statistics
    summary = {
        'channel': channel_name,
        'total_tiles_processed': len([m for m in tile_metadata if m.get('n_spots', 0) >= 0]),
        'total_tiles_attempted': len(tile_metadata),
        'total_spots': len(combined_spots_df),
        'spots_per_tile': [m.get('n_spots', 0) for m in tile_metadata],
        'average_spots_per_tile': np.mean([m.get('n_spots', 0) for m in tile_metadata]),
        'tiles_with_errors': len([m for m in tile_metadata if 'error' in m]),
    }
    
    summary_path = output_folder / f"ch_{channel_name}_processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nProcessing Summary for channel {channel_name}:")
    logger.info(f"  Tiles processed: {summary['total_tiles_processed']}/{summary['total_tiles_attempted']}")
    logger.info(f"  Total spots detected: {summary['total_spots']}")
    logger.info(f"  Average spots per tile: {summary['average_spots_per_tile']:.1f}")
    if summary['tiles_with_errors'] > 0:
        logger.warning(f"  Tiles with errors: {summary['tiles_with_errors']}")


# ============================================================================
# MAIN RUN FUNCTION
# ============================================================================

def run():
    """
    Run function for multi-tile puncta detection.
    
    This function reads a JSON manifest created by the dispatcher that specifies
    either a single tile or multiple tiles to process for a given channel.
    It processes each tile, transforms coordinates to global space if position
    information is available, and concatenates results.
    
    The function is backward compatible and will work with single-tile datasets.
    """

    # Code Ocean folders
    RESULTS_FOLDER = Path(os.path.abspath("../results"))
    DATA_FOLDER = Path(os.path.abspath("../data"))

    # Load spot channel configuration (created by dispatcher)
    spot_dict_path = list(DATA_FOLDER.glob("spot_channel_*.json"))
    
    if not len(spot_dict_path):
        raise FileNotFoundError("No spot channel dictionary was found!")

    spot_dict = utils.read_json_as_dict(spot_dict_path[0])
    spot_channel = spot_dict.get("spot_channels")

    if spot_channel is None:
        raise ValueError("Please, provide a spot channel in the dictionary")

    # NEW: Check for multi-tile processing mode
    processing_mode = spot_dict.get("processing_mode", "single_tile")
    tiles_info = spot_dict.get("tiles", [])
    n_tiles = spot_dict.get("n_tiles", 0)
    
    # Setup main output folder
    if processing_mode == "multi_tile":
        main_output_folder = RESULTS_FOLDER.joinpath(f"ch_{spot_channel}_multi_tile_spots")
    else:
        # Backward compatibility: single tile processing
        main_output_folder = RESULTS_FOLDER.joinpath(f"ch_{spot_channel}_spots")
    
    utils.create_folder(dest_dir=str(main_output_folder), verbose=True)
    
    # Create main logger
    main_logger = utils.create_logger(
        output_log_path=str(main_output_folder)
    )
    
    main_logger.info(f"Processing mode: {processing_mode}")
    main_logger.info(f"Channel: {spot_channel}")
    main_logger.info(f"Number of tiles: {n_tiles}")

    # Spot detection parameters (same for all tiles)
    sigma_zyx = [1.8, 1.0, 1.0]
    background_percentage = 25
    axis_pad = int(1.6 * max(max(sigma_zyx[1:]), sigma_zyx[0]) * 5)
    min_zyx = [3, 3, 3]
    filt_thresh = 20
    raw_thresh = 180
    context_radius = 3
    radius_confidence = 0.05
    multiscale = "0"

    # Base spot detection parameters
    base_puncta_params = {
        "multiscale": multiscale,
        "prediction_chunksize": (128, 128, 128),
        "target_size_mb": 2048,
        "n_workers": 0,
        "batch_size": 1,
        "axis_pad": axis_pad,
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

    # ========================================================================
    # MAIN PROCESSING LOGIC
    # ========================================================================
    
    if processing_mode == "multi_tile" and n_tiles > 0:
        # ====================================================================
        # MULTI-TILE PROCESSING
        # ====================================================================
        main_logger.info(f"Starting multi-tile processing for {n_tiles} tiles")
        
        # Process all tiles
        all_spots_dfs, tile_metadata = process_multi_tile_detection(
            tiles_info=tiles_info,
            base_puncta_params=base_puncta_params,
            output_folder=main_output_folder,
            main_logger=main_logger,
            multiscale=multiscale,
        )
        
        # Concatenate and save results
        concatenate_and_save_results(
            all_spots_dfs=all_spots_dfs,
            tile_metadata=tile_metadata,
            output_folder=main_output_folder,
            channel_name=spot_channel,
            logger=main_logger,
        )
        
        main_logger.info("\nMulti-tile processing complete!")
        
    else:
        # ====================================================================
        # SINGLE-TILE PROCESSING (Backward compatibility)
        # ====================================================================
        main_logger.info("Using single-tile processing mode")
        
        # Find the data file (original behavior)
        data_channels = list(DATA_FOLDER.glob(f"*{spot_channel}*.zarr")) + \
                       list(DATA_FOLDER.glob(f"*{spot_channel}*.ome.zarr"))
        
        if not len(data_channels):
            raise FileNotFoundError(
                f"No zarr files found for channel {spot_channel} in {DATA_FOLDER}"
            )
        
        data_path = data_channels[0]
        main_logger.info(f"Processing single file: {data_path}")
        
        # Run detection using original approach
        base_puncta_params['dataset_path'] = str(data_path)
        base_puncta_params['output_folder'] = main_output_folder
        base_puncta_params['logger'] = main_logger
        
        z1_puncta_detection(**base_puncta_params)
        
        main_logger.info("Single-tile processing complete!")


if __name__ == "__main__":
    run()