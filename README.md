# spot_quant
Quantify fluorescence within 3D or Z-projected masks.

# Installation

Installation instructions are given for pixi, but you can also use conda/mamba.
1. If you don't have pixi yet, install it from:`https://pixi.prefix.dev/latest/installation/`
2. Navigate to where you want to place the repository.
3. Clone the repository: `git clone https://github.com/CellFateNucOrg/spot_quant/`
4. Install the required packages: `pixi install`
5. Confirm that CUDA is available (so you can run the pipeline with GPU support):
```
srun --gres gpu:1 --pty bash
pixi run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```
This should print something like:
```
2.14.0+cu130
13.0
True # This confirms that is CUDA available
```

# Use
To do the quantification, first configure the parameters in spot_quant.sh:
* `src_dirs=()`: List of one or more directories (separated by space or line)with raw images you want to quantify.
* `filter_out=()`: Expression(s) (separated by space or line) to filter out specific files. If your input directory contains files other than your raw data, make sure to use this option to filter out those files.
* `props_c`: Which channel (starting from zero) to quantify.
* `marker_c`: Which channel was used for the segmentation (i.e., to make the masks).
* `masks_folder`: In which folder the masks are located, relative to the input directory containing the raw images.
* `mask_str`: Suffix that differentiates the mask from its corresponding image (e.g., '_mask.tif, '_seg.npy').
* `do_3d`: Whether to quantify within the full 3D mask (true or false).
* `do_mip`: Whether to quantify within a Z-projection of each mask (true or false).
* `out_folder`: Name of a subfolder in which the output is save (optional).
* `min_planes`: Filter used masks based on a minimum number of planes. (defaults to zero).
* `pixi_dir`: Directory into which you cloned this repository.

Then run the script by typing: `sbatch spot_quant.sh`.

# Output
Measure regionprops in 3D and/or based on individually Z-projected regions. Measured properties are stored in files named "props.json" and/or "mip_props.sjon", which are structured as follows:
```
[   
    {
        "image": filename (str),
        "measured_channel": value passed as 'props_c' (int),
        "min_planes": value passed as 'min_planes' (int),
        "time_points": [
            {
                "time_point": time point (int),
                "props": [
                    {
                        "label": label of the region (int),
                        "n_components": 1,
                        "bbox": [
                            minimum Z coordinate (int),
                            minimum Y coordinate (int),
                            minimum X coordinate (int),
                            maximum Z coordinate (int),
                            maximum Y coordinate (int),
                            maximum X coordinate (int)
                        ],
                        "z_index": position of the region in the output .tif file based on wh,
                        "area": regionprops area (float),
                        "diameter": regionprops equivalent_diameter (float),
                        "solidity": solidity, measured by regionprops for 3D images or calculated manually for 2D images (float),
                        "intensity_mean": mean value of all pixels in area (float),
                        "intensity_std": standard deviation of all pixel values in area (float)
                    },
                    {
                        next label...
                    },
                    ...
                ]
            },
            {
                next time point...
            },
            ...

        ]
    },
    {
        next image...
    },
    ...
]
```
Put into words: the output `.json` file contains a list of objects; each object specifies filename, which channel was measured, the minimum number of planes, and a key named `"time_points"`, which has a list containing one object per time point; each such object specifies the time point and a key named `"props"`, which has a list of which each object corresponds to a label and its measurements.

In addition, for each image, a stack containing only the measured regions (3D and/or Z-stacked projections) is saved as a .tif file.
