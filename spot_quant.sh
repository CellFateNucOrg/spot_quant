#!/bin/bash
#SBATCH --job-name=spot_quant
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem 64GB
#SBATCH --gres=gpu::1

#### Config ####

# src_dir: Directories with images to segment.
# filter_out: Expression(s) to filter out specific files.
# props_c: Which channel to quantify.
# marker_c: Channel used for the segmentation.
# masks_folder: Folder (relative to src_dir) containing the masks.
# mask_str: String used to name the mask image ( extension should be .tif, .npy or .npz).
# do_3d: Whether to output regionprops for the full mask image.
# do_mip: Whether to output regionprops for Z-projected masks (MIP).
# min_planes: Minimum height (number of planes) per mask.
# out_folder: Subfolder of in which to save the output.
# pixi_dir: Directory of the pixi workspace

src_dirs=(
)
filter_out=(faulty)
props_c=1
marker_c=0
masks_folder=dw/masks
mask_str='_dw_mask.tif'
do_3d=false
do_mip=true
out_folder=z10
min_planes=10
pixi_dir=/Volumes/meister.data/dario/code/spot_quant

#### Script ####

pixi_dir=${pixi_dir/Volumes/mnt}
cd $pixi_dir

for dir in ${src_dirs[@]}; do
    pixi run python spot_quant.py \
    --src_dir "${dir/Volumes/mnt}" \
    --filter_out "${filter_out[@]}" \
    --props_c "$props_c" \
    --marker_c "$marker_c" \
    --masks_folder "$masks_folder" \
    --mask_str "$mask_str" \
    --do_3d "$do_3d" \
    --do_mip "$do_mip" \
    --min_planes "$min_planes" \
    --out_folder "$out_folder"
done
