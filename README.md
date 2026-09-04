# spot_quant
Quantify fluorescence within 3D or Z-projected masks.

# Installation

# Use
Installation instructions are given for pixi, but you can also use conda/mamba.
1. If you don't have pixi yet, install it from: `https://pixi.prefix.dev/latest/installation/`
2. Navigate to where you want to place the repository.
3. Clone the repository: `git clone https://github.com/CellFateNucOrg/spot_quant/`.
4. Confirm that Cuda (so you can run the pipeline using a GPU) is available:
```
srun --gres gpu:1 --pty bash
pixi run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

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
Put into words: the output .json file contains a list of objects; each object specifies the filename, which channel was measured, the minimum number of planes, and the key-value pair "time_points" : []; the "time_points" list contains one object per time point, each of which specifies a time point and the key-value pair "props" : []; the "props" list contains one one object per label and its measurements).

In addition, for each image, a stack containing only the measured regions (3D and/or Z-stacked projections) is saved as a .tif file.
