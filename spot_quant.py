import argparse
from pathlib import Path
from bioio import BioImage
from bioio_ome_tiff.writers import OmeTiffWriter
# from bioio_tifffile import Reader as TiffReader
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image
from scipy.ndimage import label
import json
from utils import parse_path, collect_images


def find_max_component(region):
    """
    Find the largest component inside a region.
    """
    # Label components inside region
    components, n_components = label(region.image == 1)

    if n_components > 1:
        # Get props for all components & pick the largest one
        component_regions = regionprops(
            label_image=components,
            intensity_image=region.image_intensity
        )
        max_component = max(component_regions, key=lambda r: r.area)

        return max_component, n_components

    # Return the region & its mask if there is only one component
    else:
        return region, n_components


def get_props(regions_dict, props_c, marker_c, path, min_planes):
    """
    Measure regionprops in 2D or 3D. Returns a list with props sorted by time pointm, and saves a stack of the measured regions.
    
    Args:
        regions_dict (dict of int: [RegionProperties]): Dict with time points and regionprops (per time point) as key-value pairs.
        props_c (int): Which channel to measure.
        marker_c (int): Channel used for the segmentation.
        path (str or Path): Path for output .tif file.
        min_planes (int): Minimum number of planes per mask. Smaller regions are ignored.
    """
    # Make list for props &
    img_props = []

    # Make list for the data to stack
    img_regions = []

    # Get dtype of input data
    r0 = next(iter(regions_dict.values()))
    dtype = r0[0].image_intensity.dtype

    # Set initial min & max bbox values
    min_z = np.inf
    min_y = np.inf
    min_x = np.inf
    max_z = 0
    max_y = 0
    max_x = 0

    # Loop over time points
    for t, regions in regions_dict.items():
        # Make list for props for the current time point
        t_props = []
        t_regions = []

        # Loop over regions for current time point
        for region in regions:
            # Find largest component
            measure_region, n_components = find_max_component(region)

            # If there are multiple components...
            if n_components > 1:
                # ...translate the biggest component's bbox into full-image coordinates
                rz0, ry0, rx0, _, _, _ = region.bbox
                cz0, cy0, cx0, cz1, cy1, cx1 = measure_region.bbox
                bbox = (
                rz0 + cz0,
                ry0 + cy0,
                rx0 + cx0,
                rz0 + cz1,
                ry0 + cy1,
                rx0 + cx1
                )

            # Else use the full region's bbox
            else:
                bbox = region.bbox
            
            # Get bbox coordinates & calculate its height
            z0, y0, x0, z1, y1, x1 = bbox
            bbox_height = z1 - z0

            # Measure props if bbox ≥ min_planes
            if bbox_height >= min_planes:
                # Update min & max bbox values
                min_z = min(z0, min_z)
                min_y = min(y0, min_y)
                min_x = min(x0, min_x)
                max_z = max(z1, max_z)
                max_y = max(y1, max_y)
                max_x = max(x1, max_x)

                # Get data for the stack & set background to zero
                data_intensity = measure_region.image_intensity[..., props_c].copy()
                data_intensity[~measure_region.image] = 0
                marker_intensity = measure_region.image_intensity[..., marker_c].copy()
                marker_intensity[~measure_region.image] = 0

                # Append data to stack to the dict
                t_regions.append({
                    'label': region.label,
                    'bbox' : bbox,
                    'data_intensity': data_intensity,
                    'data_marker': marker_intensity
                })

                # Calculate solidity manually if the image is 2D (regionprops fails here)...
                if measure_region.image.shape[0] == 1:
                    hull = convex_hull_image(measure_region.image[0])
                    solidity = measure_region.area / hull.sum()
                # ...or get it from regionprops if the image is 3D
                else:
                    solidity = measure_region.solidity

                # Add props to the list for the current time_point
                t_props.append({
                    'label' : region.label,
                    'n_components': n_components,
                    'bbox': bbox,
                    'z_index': None,
                    'area': measure_region.area,
                    'diameter': measure_region.equivalent_diameter_area,
                    'solidity': solidity if np.isfinite(solidity) else None,
                    'intensity_mean': measure_region.intensity_mean[props_c],
                    'intensity_std': measure_region.intensity_std[props_c]
                })

        # Add region data
        img_regions.append({
            'time_point': t,
            'regions': t_regions
        })

        # Assign each label an index based on where its bbox starts in Z
        for i, item in enumerate(sorted(t_props, key=lambda r: r['bbox'][0])):
            item['z_index'] = i

        # Append props
        img_props.append({
            'time_point': t,
            'props': t_props
        })

    # Make a zeroes stack for the measured regions
    stack = np.zeros(
        shape=(
        len(regions_dict),
        2,
        max_z - min_z,
        max_y - min_y,
        max_x - min_x
        ),
        dtype=dtype
    )

    # Add data to the stack
    for t, t_regions in enumerate(img_regions):
        for r in t_regions['regions']:
            # Get bbox of region
            z0, y0, x0, z1, y1, x1 = r['bbox']

            # Add props_c data
            stack_slice = stack[
                t,
                props_c,
                z0 - min_z:z1 - min_z,
                y0 - min_y:y1 - min_y,
                x0 - min_x:x1 - min_x,
            ]
            data = r['data_intensity']
            stack_slice[data != 0] = data[data != 0]

            # Add marker_c data
            stack_slice = stack[
                t,
                marker_c,
                z0 - min_z:z1 - min_z,
                y0 - min_y:y1 - min_y,
                x0 - min_x:x1 - min_x,
            ]
            data = r['data_marker']
            stack_slice[data != 0] = data[data != 0]

    OmeTiffWriter.save(stack, path)
                
    return img_props


def get_mip_props(regions_dict, props_c, marker_c, path, min_planes):
    """
    Measure regionprops on Z-projected segmentations. Returns a list with props sorted by time point, and saves a stack of the projected regions.
     
     Args:
         regions_dict (dict of int: [RegionProperties]): Dict with time points and regionprops (per time point) as key-value pairs.
         props_c (int): Which channel to measure.
         marker_c (int): Channel used for the segmentation.
         path (str or Path): Path for output .tif file.
         min_planes (int): Minimum number of planes per mask. Smaller regions are ignored.
    """

    # Make lists for image props & projections
    img_mips = []
    img_props = []

    # Get dtype of input data
    r0 = next(iter(regions_dict.values()))
    dtype = r0[0].image_intensity.dtype

    # Set bbox_max to zero initially
    bbox_max = 0

    # Loop over time points
    for t, regions in regions_dict.items():
        # Make list for time point props & projections
        t_mips = []
        t_props = []
        
        # Loop over regions for current time point
        for region in regions:
            # Find largest component
            measure_region, n_components = find_max_component(region)

            # If there are multiple components...
            if n_components > 1:
                # ...translate the biggest component's bbox to full-image coordinates
                rz0, ry0, rx0, _, _, _ = region.bbox
                cz0, cy0, cx0, cz1, cy1, cx1 = measure_region.bbox
                bbox = (
                rz0 + cz0,
                ry0 + cy0,
                rx0 + cx0,
                rz0 + cz1,
                ry0 + cy1,
                rx0 + cx1
                )

            # Else use the full region's bbox
            else:
                bbox = region.bbox
            
            # Get bbox coordinates & calculate its height
            z0, y0, x0, z1, y1, x1 = bbox
            bbox_height = z1 - z0

            # Project region & measure props if mask is larger min_planes
            if bbox_height >= min_planes:
                # Find current largest bbox length
                bbox_max = max(y1 - y0, x1 - x0, bbox_max)
                
                # Project intensity image
                intensity_region = measure_region.image_intensity[..., props_c].copy()
                intensity_region[~measure_region.image] = 0
                intensity_mip = intensity_region.max(axis=0)

                marker_region = measure_region.image_intensity[..., marker_c].copy()
                marker_region[~measure_region.image] = 0
                marker_mip = marker_region.max(axis=0)

                # Append projections
                t_mips.append({
                    'z_index': None,
                    'label': region.label,
                    'bbox': bbox,
                    'data_intensity': intensity_mip,
                    'data_marker': marker_mip,

                })
                # Measure props for the projected region
                projected_region = regionprops(
                    label_image=measure_region.image.max(axis=0).astype(np.uint8),
                    intensity_image=intensity_mip
                )[0]

                # Append props
                solidity = projected_region.solidity
                t_props.append({
                    'label': region.label,
                    'n_components': n_components,
                    'bbox': bbox,
                    'z_index': None,
                    'area': projected_region.area,
                    'diameter': projected_region.equivalent_diameter_area,
                    'solidity': solidity if np.isfinite(solidity) else None,
                    'intensity_mean': projected_region.intensity_mean,
                    'intensity_std': projected_region.intensity_std
                })

        # Assign each label an index based on where its bbox starts in Z
        for i, item in enumerate(sorted(t_mips, key=lambda r: r['bbox'][0])):
            item['z_index'] = i

        for i, item in enumerate(sorted(t_props, key=lambda r: r['bbox'][0])):
            item['z_index'] = i

        # Append props
        img_mips.append({
            'time_point': t,
            'mips': t_mips
        })

        img_props.append({
            'time_point': t,
            'props': t_props
        })        

    # Make a zeroes array for stacking the projections in Z
    max_n_time_points = len(regions_dict)
    max_n_mips = max(len(t['mips']) for t in img_mips)
    stack = np.zeros(
        shape=(
            max_n_time_points,
            2,
            max_n_mips,
            bbox_max,
            bbox_max
        ),
        dtype=dtype
    )

    # Add projections to the stack
    for t, t_mips in enumerate(img_mips):
        for p in t_mips['mips']:
            # Find centre position for the projected region
            y_length = p['bbox'][4] - p['bbox'][1]
            y_diff = bbox_max - y_length
            y_min = y_diff // 2
            y_max = y_min + y_length

            x_length = p['bbox'][5] - p['bbox'][2]
            x_diff = bbox_max - x_length
            x_min = x_diff // 2
            x_max = x_min + x_length

            # Add projections to the zeroes stack
            stack[
                t,
                props_c,
                p['z_index'],
                y_min:y_max, 
                x_min:x_max
                ] = p['data_intensity']
            
            stack[
                t,
                marker_c,
                p['z_index'],
                y_min:y_max, 
                x_min:x_max
                ] = p['data_marker']

    OmeTiffWriter.save(stack, path)

    return img_props


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', required=True, help='Folder with images to segment.')
    parser.add_argument('--filter_out', required=True, nargs='*', help='Expression(s) to filter out specific files.')
    parser.add_argument('--props_c', required=True, type=int, help='Which channel to quantify.')
    parser.add_argument('--marker_c', required=True, type=int, help='Channel used for the segmentation.')
    parser.add_argument('--masks_folder', required=True, help='Folder (relative to src_dir) containing the masks.')
    parser.add_argument('--mask_str', required=True, help='String used to name the mask image ( extension should be .tif, .npy or .npz).')
    parser.add_argument('--do_3d', required=True, help='Whether to output regionprops for the full mask image.')
    parser.add_argument('--do_mip', required=True, help='Whether to output regionprops for Z-projected masks (MIP).')
    parser.add_argument('--min_planes', required=False, default=0, type=int, help='Minimum height (number of planes) per mask.')
    parser.add_argument('--out_folder', required=False, help='Subfolder of in which to save the output.')
    return parser.parse_args()

def main():
    args = get_args()
    src_dir = Path(args.src_dir)
    filter_out = [e for e in args.filter_out]
    props_c = args.props_c
    marker_c = args.marker_c
    masks_folder = args.masks_folder
    mask_str = args.mask_str
    do_3d = args.do_3d.lower() in ('true', 1, 'yes')
    do_mip = args.do_mip.lower() in ('true', 1, 'yes') 
    min_planes = args.min_planes
    out_folder = args.out_folder

    # Collect images
    imgs = collect_images(parse_path(src_dir), filter_out=filter_out)

    # Make path for output
    out_dir = src_dir / f'spot_quant/{out_folder}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make lists for props
    props = []
    props_mip = []

    for img in imgs:
        print(f'Processing image {img.name}')

        # Make paths for output
        path_json = out_dir / f'props.json'
        path_tif = out_dir / f'{img.stem}_regions.tif'
        path_mip_json = out_dir / f'props_mip.json'
        path_mip_tif = out_dir / f'{img.stem}_regions_max.tif'

        # Read image & mask
        load_img = BioImage(img)
        mask_path = img.parent / f'{masks_folder}/{img.stem}{mask_str}'
        load_mask = BioImage(mask_path)

        # Make a dict with props for each time point
        regions_dict = {}
        for t in range(load_img.dims['T'][0]):
            # Get image data for current time point and move the C axis to the end
            img_data = load_img.get_image_data('TCZYX', T=t).squeeze()
            img_data = np.moveaxis(img_data, 0, -1)
            mask_data = load_mask.get_image_data('TCZYX', T=t, C=0).squeeze()

            # Add data to the dict
            regions_dict[t] = regionprops(
                label_image=mask_data,
                intensity_image=img_data
            )

        # Add props to lists (and save a stack of the measured regions)
        if do_3d:
            props.append({
                'image': img.name,
                'measured_channel': props_c,
                'min_planes': min_planes,
                'time_points': get_props(
                    regions_dict=regions_dict,
                    props_c=props_c,
                    marker_c=marker_c,
                    path=path_tif,
                    min_planes=min_planes
                )
            })

        if do_mip:
            props_mip.append({
                'image': img.name,
                'measured_channel': props_c,
                'min_planes': min_planes,
                'time_points': get_mip_props(
                    regions_dict=regions_dict,
                    props_c=props_c,
                    marker_c=marker_c,
                    path=path_mip_tif,
                    min_planes=min_planes
                )
            })

    # Save props as .json
    if do_3d:
        with open(path_json, 'w') as f:
            json.dump(props, f, indent=4)
        print(f'Saved measurements of input images as {path_json}')

    if do_mip:
        with open(path_mip_json, 'w') as f:
            json.dump(props_mip, f, indent=4)
        print(f'Saved measurements of projected images as {path_mip_json}')


if __name__=='__main__':
    main()
