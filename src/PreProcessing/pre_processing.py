import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
# from dipy.align.reslice import reslice
from src.Utils.volume_utilities import intensity_normalization, resize_volume, crop_MR
from src.Utils.io import load_nifti_volume


def run_pre_processing(filename, pre_processing_parameters):
    print("Extracting data...")
    #@TODO. Only nifti input or also accomodate for other types? Or maybe just re-export to nifti with SimpleITK?
    input_ext = os.path.splitext(filename)

    nib_volume = load_nifti_volume(filename)

    print("Pre-processing...")
    # Normalize spacing
    new_spacing = pre_processing_parameters.output_spacing
    if pre_processing_parameters.output_spacing is None:
        tmp = np.min(nib_volume.header.get_zooms())
        new_spacing = [tmp, tmp, tmp]

    library = pre_processing_parameters.preprocessing_library
    if library == 'nibabel':
        resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
        data = resampled_volume.get_data().astype('float32')
    # elif library == 'dipy':
    #     data, aff = reslice(nib_volume.get_data(), nib_volume.affine, nib_volume.header.get_zooms(), new_zooms=new_spacing)
    #     resampled_volume = nib.Nifti1Image(data, affine=aff)

    crop_bbox = None
    # Normalize values
    data = intensity_normalization(volume=data, parameters=pre_processing_parameters)
    # Exclude background
    if pre_processing_parameters.crop_background:
        data, crop_bbox = crop_MR(data)

    data = resize_volume(data, pre_processing_parameters.new_axial_size, pre_processing_parameters.slicing_plane,
                         order=1)

    return nib_volume, resampled_volume, data, crop_bbox
