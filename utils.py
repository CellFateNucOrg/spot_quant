from pathlib import Path
from bioio import BioImage
from bioio_ome_tiff.writers import OmeTiffWriter
import numpy as np


def parse_path(path, server=True):
    """
    Return a Path object and replaces 'Volumes' with 'mnt' if server is True.
    """
    if server:
        return Path(str(path).replace('Volumes', 'mnt'))
    else:
        return Path(path)


def collect_images(src_dir, ftypes=('.tif', '.nd2', '.czi', '.lif'), filter_in=(), filter_out=()):
    imgs = []
    for item in Path(src_dir).iterdir():
        if item.suffix not in ftypes:
            continue
        if not all(f in item.name for f in filter_in):
            continue
        if any(f in item.name for f in filter_out):
            continue
        imgs.append(item)
    return imgs


def split_stack(imgs, planes, output_dir, channel=0):
    """
    Extract substacks or slices from one or more stacks.

    Args:
        imgs (list[str], list[Path]): List of image paths.
        planes (list[int] or range): Planes (ints) or substack (range) to extract.
        output_dir (str, Path): Where to store the extracted slices.
        channel (int): Channel to extract. Default 0.

    Returns:
        List with paths of the extracted slices.
    """

    out = []
    for img in imgs:
        # Extract substack if 'planes' is a range
        if isinstance(planes, range):
           suffix = f'{img.stem}_c{str(channel).zfill(3)}_z{planes[0]}-{planes[-1]-1}'
           path = Path(output_dir) / f'{suffix}.tif' 
           out.append(path)
           data = BioImage(img).get_image_data('TCZYX', C=channel, Z=planes)
           OmeTiffWriter.save(data, path)

        # Extract slices if 'planes' is a list of ints
        elif isinstance(planes, list) and all(isinstance(zi, int) for zi in planes):
            for z in planes:
                suffix = f'{img.stem}_c{str(channel).zfill(3)}_z{str(z).zfill(3)}'
                path = Path(output_dir) / f'{suffix}.tif'
                out.append(path)
                data = BioImage(img).get_image_data('TCZYX', C=channel, Z=z)
                OmeTiffWriter.save(data, path)

    return out


def save_masks(masks, mask_path):
    """
    Save mask as .tif, .npy, or .npz.
    """
    if mask_path.suffix == '.tif':
        OmeTiffWriter.save(masks, mask_path)
    elif mask_path.suffix == '.npy':
        np.save(mask_path, masks)
    elif mask_path.suffix == '.npz':
        np.savez(mask_path, masks)
    else:
        raise ValueError(f'Unsupported mask format: {mask_path.suffix}. Use .tif, .npy, or .npz.')
