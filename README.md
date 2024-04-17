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

## Documentation
You can access the documentation for this module [here]().

## Contributing

To develop the code, install the packages described in the Dockerfile.

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```
flake8 . --max-line-length=100
```

- Use **black** to automatically format the code into PEP standards:
```
black .
```

- Use **isort** to automatically sort import statements:
```
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repo and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect the build system or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests