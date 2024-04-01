# aind-z1-spot-detection

Large-scale detection of spot-like structures. The algorithms included in this package are:

**Traditional algorithm**: Method based on the Laplacian of Gaussians technique. These are the image processing steps that happen in every chunk:
1. Laplacian of Gaussians to enhance regions where the intensity changes dramatically (higher gradient).
2. Percentile to get estimated background image.
3. Combination of logical ANDs to filter the LoG image using threshold values and non-linear maximum filter.
4. After identifying initial spots (ZYX points) from 1-3 steps, we prune blobs close to each other within a certain radius $$r$$ using a kd-tree.
5. We take each of these pruned spots and get their contexts which is a 3D image of size $$context = (radius + 1, radius + 1, radius + 1)$$.
6. Finally, we fit a gaussian to each of the spots using its context to be able to prune false positives.

This is a traditional-based algorithm, therefore, parameter tunning is required to make it work.

## Optional mask
An additional segmentation mask can be provided to only detect spots in given areas. In our case, the segmentation mask provided is a cell segmentation mask of the entire dataset, we used a  [large-scale version of cellpose](https://github.com/AllenNeuralDynamics/aind-z1-cell-segmentation) to generate the segmentation masks.
When segmentation masks are provided, the `spots.npy` file that is created in the output directory will contain the segmentation ID of that detected spot for further post-processing and analysis. The segmentation ID will be in the last position of the array, e.g., [..., [Z, Y, X, SEG_ID], ...].