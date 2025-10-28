import dask.array as da
import dask_image.ndinterp
import numpy as np
import boto3
import xmltodict
from scipy import ndimage
from collections import defaultdict, OrderedDict


def apply_camera_alignment_to_tile_array(tile_array: da.array, tile_name: str, xml_path: str): 

    transforms = None
    if xml_path:
        try:
            transforms = load_and_calculate_transforms(xml_path)
            print(f"Loaded transforms for {len(transforms)} tiles from XML")
        except Exception as e:
            print(f"Warning: Could not load transforms from XML: {e}")


    tile_array = apply_transform_to_tile(tile_array, tile_name, transforms, xml_path)
    chunk_size = (1,1,128,128,128)
    tile_array=tile_array.rechunk(chunk_size)
    return tile_array


def load_xml(xml_path)-> OrderedDict:
    # Check if it's an S3 path
    if str(xml_path).startswith('s3://'):
        # Parse S3 path
        s3_path = xml_path.replace("s3://", "")
        bucket_name, key = s3_path.split("/", 1) 
        
        # Read from S3
        client = boto3.client('s3')
        try:
            response = client.get_object(Bucket=bucket_name, Key=key)
            xml_content = response['Body'].read().decode('utf-8')
        except Exception as e:
            raise Exception(f"Could not read XML file from S3: {e}")
    else:
        # Read from local file
        with open(xml_path, "r") as file:
            xml_content = file.read()
    
    # Parse XML content
    data: OrderedDict = xmltodict.parse(xml_content)
    return data

def get_tile_id_from_name(data:dict, tilename):
    """
    """
    
    #find viewsetup with matching tilename
    viewsetups = data["SpimData"]["SequenceDescription"][
            "ViewSetups"
        ]["ViewSetup"]
    matching_viewsetup=[v for v in viewsetups if tilename in v['name']]

    #get tile_number from this viewsetup
    matching_tile_number = matching_viewsetup[0]['attributes']['tile']

    return int(matching_tile_number)
def extract_tile_name_from_path(tile_path: str) -> str:
    """
    Extract tile name from the file path for matching with XML transforms.
    
    Parameters
    ----------
    tile_path : str
        Full path to tile file
        
    Returns
    -------
    str
        Extracted tile name
    """
    # Extract filename from path
    filename = tile_path.rstrip('/').split('/')[-1]
    # Remove .ome.zarr extension
    tile_name = filename.replace('.ome.zarr', '').replace('.ome.zarr/', '')
    return tile_name

def extract_tile_transforms(xml_path: str) -> dict[int, list[dict]]:
    """
    Parses BDV xml and outputs map of setup_id -> list of transformations
    Output dictionary maps view number to list of {'@type', 'Name', 'affine'}
    where 'affine' contains the transform as string of 12 floats.

    Matrices are listed in the order of forward execution.

    Parameters
    ------------------------
    xml_path: str
        Path of xml outputted by BigStitcher. Can be local path or S3 path (s3://bucket/path/file.xml).

    Returns
    ------------------------
    dict[int, list[dict]]
        Dictionary of tile ids to transform list. List entries described above.

    """

    view_transforms: dict[int, list[dict]] = {}
    
    data = load_xml(xml_path)
    
    view_registration = data["SpimData"]["ViewRegistrations"]["ViewRegistration"]
    if not isinstance(view_registration, list):
        tfm_stack = view_registration["ViewTransform"]
        
        if type(tfm_stack) is not list:
            tfm_stack = [tfm_stack]
        view_transforms[int(view_registration["@setup"])] = tfm_stack
    else:
        for view_reg in view_registration:
            tfm_stack = view_reg["ViewTransform"]
            if type(tfm_stack) is not list:
                tfm_stack = [tfm_stack]
            view_transforms[int(view_reg["@setup"])] = tfm_stack

    view_transforms = {
        view: tfs[::-1] for view, tfs in view_transforms.items()
    }

    return view_transforms


def extract_second_affine_transforms_raw(xml_path: str) -> dict[int, list[dict]]:
    """
    Extract only the second affine transformation from each tile's transform stack in raw format.
    This returns the raw dictionary format compatible with calculate_net_transforms().
    
    Parameters
    ----------
    xml_path : str
        Path to XML file. Can be local path or S3 path (s3://bucket/path/file.xml).
        
    Returns
    -------
    dict[int, list[dict]]
        Dictionary mapping tile IDs to their second affine transform in raw format
    """
    # Get all transforms in raw format
    view_transforms = extract_tile_transforms(xml_path)
    
    second_transforms_raw = {}
    
    for view_id, transform_list in view_transforms.items():
        # Look for the second transform in the list
        if len(transform_list) >= 2:
            # Get only the second transform as a single-item list
            second_transform = transform_list[1]  # Index 1 for second transform
            second_transforms_raw[view_id] = [second_transform]
        else:
            # If no second transform, create identity transform in raw format
            identity_raw = {
                "@type": "affine",
                "Name": "identity", 
                "affine": "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0"
            }
            second_transforms_raw[view_id] = [identity_raw]
            print(f"Warning: No second transform found for tile {view_id}, using identity")
    
    return second_transforms_raw


def calculate_net_transforms(
    view_transforms: dict[int, list[dict]]
) -> dict[int, np.ndarray]:
    """
    Accumulate net transform and net translation for each matrix stack.
    Net translation =
        Sum of translation vectors converted into original nominal basis
    Net transform =
        Product of 3x3 matrices
    NOTE: Translational component (last column) is defined
          wrt to the DOMAIN, not codomain.
          Implementation is informed by this given.

    Parameters
    ------------------------
    view_transforms: dict[int, list[dict]]
        Dictionary of tile ids to transforms associated with each tile.

    Returns
    ------------------------
    dict[int, np.ndarray]:
        Dictionary of tile ids to net transform.

    """

    identity_transform = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    net_transforms: dict[int, np.ndarray] = defaultdict(
        lambda: np.copy(identity_transform)
    )

    for view, tfs in view_transforms.items():
        net_translation = np.zeros(3)
        net_matrix_3x3 = np.eye(3)
        curr_inverse = np.eye(3)

        for (tf) in (tfs):  # Tfs is a list of dicts containing transform under 'affine' key
            nums = [float(val) for val in tf["affine"].split(" ")]
            matrix_3x3 = np.array([nums[0::4], nums[1::4], nums[2::4]])
            translation = np.array(nums[3::4])
            
            # print(translation)
            nums = np.array(nums).reshape(3,4)
            matrix_3x3 = np.array([nums[:,0], nums[:,1], nums[:,2]]).T
            translation = np.array(nums[:,3])
            
            #old way
            net_translation = net_translation + (curr_inverse @ translation)
            net_matrix_3x3 = matrix_3x3 @ net_matrix_3x3  
            curr_inverse = np.linalg.inv(net_matrix_3x3)  # Update curr_inverse

        net_transforms[view] = np.hstack(
            (net_matrix_3x3, net_translation.reshape(3, 1))
        )

    return net_transforms

def load_and_calculate_transforms(xml_path: str) -> dict:
    """
    Load XML file and calculate net transforms for all tiles.
    
    Parameters
    ----------
    xml_path : str
        Path to XML file containing transforms
        
    Returns
    -------
    dict
        Dictionary mapping tile names/IDs to net transforms
    """
    # Extract raw transforms from XML
    view_transforms = extract_second_affine_transforms_raw(xml_path)
    
    # Calculate net transforms - only necessary if accumulating multiple transforms
    net_transforms = calculate_net_transforms(view_transforms)
    
    return net_transforms

def apply_transform_to_tile(tile_array: da.Array, tile_name: str, transforms: dict, xml_path: str) -> da.Array:
    """
    Apply affine transform to a dask array tile using dask_image.
    
    Parameters
    ----------
    tile_array : da.Array
        Dask array containing tile data (5D: T,C,Z,Y,X)
    tile_name : str
        Name of the tile for transform lookup
    transforms : dict
        Dictionary of transforms

    Note: BigStitcher XML uses XYZ mode, scipy uses ZYX mode with backward transforms
    """
    def convert_xyz_mat_to_zyx(affine_mat):
        """Convert 4x4 affine from XYZ to ZYX coordinate ordering"""
        shuffled_rows = affine_mat[[2, 1, 0, 3], :]
        shuffled_cols = shuffled_rows[:, [2, 1, 0, 3]]
        return shuffled_cols
    
    print(f'tile name {tile_name}')
    print(f'transforms {transforms}')

    # Find the appropriate transform for this tile
    if tile_name in transforms:
        transform_matrix = transforms[tile_name]
    else:
        data = load_xml(xml_path)
        tile_id = get_tile_id_from_name(data, tile_name)
        transform_matrix = transforms[tile_id]
    
    if transform_matrix is None:
        print(f"Warning: No transform found for tile {tile_name}")
        return tile_array
    
    print(f"Applying transform to tile {tile_name}")
    
    # Reorder XYZ to ZYX
    transform_matrix = np.vstack([transform_matrix, [0, 0, 0, 1]])
    transform_matrix_reordered = convert_xyz_mat_to_zyx(transform_matrix)
    
    # For inversion, we need to work with the full 4x4 homogeneous form
    # Convert 3x4 to 4x4 by adding [0, 0, 0, 1] row
    

    # Invert for scipy/dask affine_transform (uses backward transform)
    # transform_matrix_inv = np.linalg.inv(transform_matrix_reordered)

    
    # Extract 3D transform components (not 2D!)
    matrix_3d = transform_matrix_reordered[:3, :3]  # 3x3 linear transform
    offset_3d = transform_matrix_reordered[:3, 3]   # 3D translation vector
    
    # Extract 3D volume from 5D array
    volume_3d = tile_array[0, 0, :, :, :]  # (Z,Y,X)
    
    # Apply 3D affine transform to entire volume at once
    transformed_3d = dask_image.ndinterp.affine_transform(
        volume_3d,
        matrix=matrix_3d,
        offset=offset_3d,
        order=1,  # linear interpolation (or 0 for nearest, 3 for cubic)
        mode='constant',
        cval=0.0
    )
    
    # Restore 5D shape
    transformed_5d = transformed_3d[np.newaxis, np.newaxis, :, :, :]
    
    return transformed_5d