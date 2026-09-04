from pathlib import Path


__all__ = [
    'parse_path',
    'collect_images'
]


def parse_path(path, server=True):
    """
    Return a Path object and replaces 'Volumes' with 'mnt' if server is True.
    """
    if server:
        return Path(str(path).replace('Volumes', 'mnt'))
    else:
        return Path(path)


def collect_images(src_dir, ftypes=['.tif','.nd2','.czi','.lif'], filter_in=[], filter_out=[]):
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