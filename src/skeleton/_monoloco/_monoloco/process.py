import os 
import yaml
import torch

from .utils import get_keypoints, pixel_to_camera, unnormalize_bi, to_cartesian, back_correct_angles

Sx = 7.2
Sy = 5.4

def preprocess_pifpaf(annotations, im_size=None, min_conf=0.):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []
    enlarge = 2  # Avoid enlarge boxes for social distancing

    for dic in annotations:
        kps = prepare_pif_kps(dic['keypoints'])
        box = dic['bbox']
        try:
            conf = dic['score']
            # Enlarge boxes
            delta_h = (box[3]) / (10 * enlarge)
            delta_w = (box[2]) / (5 * enlarge)
            # from width height to corners
            box[2] += box[0]
            box[3] += box[1]

        except KeyError:
            all_confs = np.array(kps[2])
            score_weights = np.ones(17)
            score_weights[:3] = 3.0
            score_weights[5:] = 0.1
            # conf = np.sum(score_weights * np.sort(all_confs)[::-1])
            conf = float(np.mean(all_confs))
            # Add 15% for y and 20% for x
            delta_h = (box[3] - box[1]) / (7 * enlarge)
            delta_w = (box[2] - box[0]) / (3.5 * enlarge)
            assert delta_h > -5 and delta_w > -5, "Bounding box <=0"

        box[0] -= delta_w
        box[1] -= delta_h
        box[2] += delta_w
        box[3] += delta_h

        # Put the box inside the image
        if im_size is not None:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(box[2], im_size[0])
            box[3] = min(box[3], im_size[1])

        if conf >= min_conf:
            box.append(conf)
            boxes.append(box)
            keypoints.append(kps)

    return boxes, keypoints


def prepare_pif_kps(kps_in):
    """Convert from a list of 51 to a list of 3, 17"""

    assert len(kps_in) % 3 == 0, "keypoints expected as a multiple of 3"
    xxs = kps_in[0:][::3]
    yys = kps_in[1:][::3]  # from offset 1 every 3
    ccs = kps_in[2:][::3]

    return [xxs, yys, ccs]

def preprocess_monoloco(keypoints, kk):

    """ Preprocess batches of inputs
    keypoints = torch tensors of (m, 3, 17)  or list [3,17]
    Outputs =  torch tensors of (m, 34) in meters normalized (z=1) and zero-centered using the center of the box
    """
    if isinstance(keypoints, list):
        keypoints = torch.tensor(keypoints)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    # Projection in normalized image coordinates and zero-center with the center of the bounding box
    uv_center = get_keypoints(keypoints, mode='center')
    
    xy1_center = pixel_to_camera(uv_center, kk, 10)
    xy1_all = pixel_to_camera(keypoints[:, 0:2, :], kk, 10)

    kps_norm = xy1_all
    kps_out = kps_norm[:, :, 0:2].reshape(kps_norm.size()[0], -1)  # no contiguous for view
    # kps_out = torch.cat((kps_out, keypoints[:, 2, :]), dim=1)
    return kps_out

def extract_outputs(outputs, tasks=()):
    """
    Extract the outputs for multi-task training and predictions
    Inputs:
        tensor (m, 10) or (m,9) if monoloco
    Outputs:
         - if tasks are provided return ordered list of raw tensors
         - else return a dictionary with processed outputs
    """
    dic_out = {'x': outputs[:, 0:1],
               'y': outputs[:, 1:2],
               'd': outputs[:, 2:4],
               'h': outputs[:, 4:5],
               'w': outputs[:, 5:6],
               'l': outputs[:, 6:7],
               'ori': outputs[:, 7:9]}

    if outputs.shape[1] == 10:
        dic_out['aux'] = outputs[:, 9:10]

    # Multi-task training
    if len(tasks) >= 1:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_out[task] for task in tasks]

    # Preprocess the tensor
    # AV_H, AV_W, AV_L, HWL_STD = 1.72, 0.75, 0.68, 0.1
    bi = unnormalize_bi(dic_out['d'])
    dic_out['bi'] = bi

    dic_out = {key: el.detach().cpu() for key, el in dic_out.items()}
    x = to_cartesian(outputs[:, 0:3].detach().cpu(), mode='x')
    y = to_cartesian(outputs[:, 0:3].detach().cpu(), mode='y')
    d = dic_out['d'][:, 0:1]
    z = torch.sqrt(d**2 - x**2 - y**2)
    dic_out['xyzd'] = torch.cat((x, y, z, d), dim=1)
    dic_out.pop('d')
    dic_out.pop('x')
    dic_out.pop('y')
    dic_out['d'] = d

    yaw_pred = torch.atan2(dic_out['ori'][:, 0:1], dic_out['ori'][:, 1:2])
    yaw_orig = back_correct_angles(yaw_pred, dic_out['xyzd'][:, 0:3])
    dic_out['yaw'] = (yaw_pred, yaw_orig)  # alpha, ry

    if outputs.shape[1] == 10:
        dic_out['aux'] = torch.sigmoid(dic_out['aux'])
    return dic_out

    
def load_calibration(im_size, focal_length= 5.7):

    kk = [
        [im_size[0]*focal_length/Sx, 0., im_size[0]/2],
        [0., im_size[1]*focal_length/Sy, im_size[1]/2],
        [0., 0., 1.]
    ]

    return kk