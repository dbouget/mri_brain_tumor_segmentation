import torch
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
from os.path import join
import numpy as np
import sys
from shutil import copy
from math import ceil, floor
from copy import deepcopy
from src.Models.PLS.PLS import PLS
from src.Utils.volume_utilities import padding_for_inference, padding_for_inference_both_ends
from src.Models.UNet.DualAttentionUNet import PAM, CAM


def run_predictions(data, model_path, parameters):
    """
    Only the prediction is done in this function, possible thresholdings and re-sampling are not included here.
    :param data:
    :return:
    """
    if model_path.split('.')[-1] == 'ckpt':
        return __run_predictions_pytorch(data, model_path)
    else:
        return __run_predictions_tensorflow(data, model_path, parameters)


def __run_predictions_pytorch(data, model_path):
    net = PLS.load_from_checkpoint(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.freeze()
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm3d):
            m.track_running_stats = False
    net.to(device)

    img = np.expand_dims(data, axis=0)
    _img = torch.from_numpy(img)
    _img = _img.unsqueeze(0)
    _img = _img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        probs = net(_img)
        probs = probs.squeeze(0)
        probs = probs.cpu()
        mask = probs.cpu().numpy()

    final_result = np.moveaxis(mask, 0, -1)

    return final_result


def __run_predictions_tensorflow(data, model_path, parameters):
    print("Loading model...")
    model = load_model(model_path, custom_objects={'PAM': PAM, 'CAM': CAM}, compile=False)

    whole_input_at_once = False
    if len(parameters.new_axial_size) == 3:
        whole_input_at_once = True

    final_result = None

    print("Predicting...")
    if whole_input_at_once:
        final_result = __run_predictions_whole(data=data, model=model,
                                               deep_supervision=parameters.training_deep_supervision)
    else:
        final_result = __run_predictions_slabbed(data=data, model=model, parameters=parameters)

    return final_result


def __run_predictions_whole(data, model, deep_supervision=False):
    data_prep = np.expand_dims(data, axis=0)
    data_prep = np.expand_dims(data_prep, axis=-1)
    predictions = model.predict(data_prep)

    if deep_supervision:
        return predictions[0][0]
    else:
        return predictions[0]


def __run_predictions_slabbed(data, model, parameters):
    """
    Working/tested for the axial and sagittal planes.
    """
    slicing_plane = parameters.slicing_plane
    slab_size = parameters.training_slab_size
    new_axial_size = parameters.new_axial_size

    upper_boundary = data.shape[2]
    if slicing_plane == 'sagittal':
        upper_boundary = data.shape[0]
    elif slicing_plane == 'coronal':
        upper_boundary = data.shape[1]

    # Placeholder for the final predictions
    final_result = np.zeros(data.shape + (parameters.training_nb_classes,))
    data = np.expand_dims(data, axis=-1)
    count = 0

    if parameters.predictions_non_overlapping:
        data, pad_value = padding_for_inference(data=data, slab_size=slab_size, slicing_plane=slicing_plane)
        scale = ceil(upper_boundary / slab_size)
        unpad = False
        for chunk in range(scale):
            if chunk == scale-1 and pad_value != 0:
                unpad = True

            if slicing_plane == 'axial':
                slab_CT = data[:, :, int(chunk * slab_size):int((chunk + 1) * slab_size), 0]
            elif slicing_plane == 'sagittal':
                tmp = data[int(chunk * slab_size):int((chunk + 1) * slab_size), :, :, 0]
                slab_CT = tmp.transpose((1, 2, 0))
            elif slicing_plane == 'coronal':
                tmp = data[:, int(chunk * slab_size):int((chunk + 1) * slab_size), :, 0]
                slab_CT = tmp.transpose((0, 2, 1))

            slab_CT = np.expand_dims(np.expand_dims(slab_CT, axis=0), axis=-1)
            slab_CT_pred = model.predict(slab_CT)

            if not unpad:
                for c in range(0, slab_CT_pred.shape[-1]):
                    if slicing_plane == 'axial':
                        final_result[:, :, int(chunk * slab_size):int((chunk + 1) * slab_size), c] = \
                            slab_CT_pred[0][:, :, :slab_size, c]
                    elif slicing_plane == 'sagittal':
                        final_result[int(chunk * slab_size):int((chunk + 1) * slab_size), :, :, c] = \
                            slab_CT_pred[0][:, :, :slab_size, c].transpose((2, 0, 1))
                    elif slicing_plane == 'coronal':
                        final_result[:, int(chunk * slab_size):int((chunk + 1) * slab_size), :, c] = \
                            slab_CT_pred[0][:, :, :slab_size, c].transpose((0, 2, 1))
            else:
                for c in range(0, slab_CT_pred.shape[-1]):
                    if slicing_plane == 'axial':
                        final_result[:, :, int(chunk * slab_size):, c] = \
                            slab_CT_pred[0][:, :, :slab_size-pad_value, c]
                    elif slicing_plane == 'sagittal':
                        final_result[int(chunk * slab_size):, :, :, c] = \
                            slab_CT_pred[0][:, :, :slab_size-pad_value, c].transpose((2, 0, 1))
                    elif slicing_plane == 'coronal':
                        final_result[:, int(chunk * slab_size):, :, c] = \
                            slab_CT_pred[0][:, :, :slab_size-pad_value, c].transpose((0, 2, 1))

            print(count)
            count = count + 1
    else:
        if slab_size == 1:
            for slice in range(0, data.shape[2]):
                slab_CT = data[:, :, slice, 0]
                if np.sum(slab_CT > 0.1) == 0:
                    continue
                slab_CT_pred = model.predict(np.reshape(slab_CT, (1, new_axial_size[0], new_axial_size[1], 1)))
                for c in range(0, slab_CT_pred.shape[-1]):
                    final_result[:, :, slice, c] = slab_CT_pred[:, :, c]
        else:
            data = padding_for_inference_both_ends(data=data, slab_size=slab_size, slicing_plane=slicing_plane)
            half_slab_size = int(slab_size / 2)
            for slice in range(half_slab_size, upper_boundary):
                if slicing_plane == 'axial':
                    slab_CT = data[:, :, slice - half_slab_size:slice + half_slab_size, 0]
                elif slicing_plane == 'sagittal':
                    slab_CT = data[slice - half_slab_size:slice + half_slab_size, :, :, 0]
                    slab_CT = slab_CT.transpose((1, 2, 0))
                elif slicing_plane == 'coronal':
                    slab_CT = data[:, slice - half_slab_size:slice + half_slab_size, :, 0]
                    slab_CT = slab_CT.transpose((0, 2, 1))

                slab_CT = np.reshape(slab_CT, (1, new_axial_size[0], new_axial_size[1], slab_size, 1))
                if np.sum(slab_CT > 0.1) == 0:
                    continue

                slab_CT_pred = model.predict(slab_CT)

                for c in range(0, slab_CT_pred.shape[-1]):
                    if slicing_plane == 'axial':
                        final_result[:, :, slice - half_slab_size, c] = slab_CT_pred[0][:, :, half_slab_size, c]
                    elif slicing_plane == 'sagittal':
                        final_result[slice, :, :, c] = slab_CT_pred[0][:, :, half_slab_size, c]
                    elif slicing_plane == 'coronal':
                        final_result[:, slice, :, c] = slab_CT_pred[0][:, :, half_slab_size, c]

                print(count)
                count = count + 1

    return final_result
