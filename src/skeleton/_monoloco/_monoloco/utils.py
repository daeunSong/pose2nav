import math 
import numpy as np
import torch
import torch.nn.functional as F


def unnormalize_bi(loc):
    """
    Unnormalize relative bi of a nunmpy array
    Input --> tensor of (m, 2)
    """
    assert loc.size()[1] == 2, "size of the output tensor should be (m, 2)"
    bi = torch.exp(loc[:, 1:2]) * loc[:, 0:1]

    return bi

def pixel_to_camera(uv_tensor, kk, z_met):
    """
    Convert a tensor in pixel coordinate to absolute camera coordinates
    It accepts lists or torch/numpy tensors of (m, 2) or (m, x, 2)
    where x is the number of keypoints
    """
    if isinstance(uv_tensor, (list, np.ndarray)):
        uv_tensor = torch.tensor(uv_tensor)
    if isinstance(kk, list):
        kk = torch.tensor(kk)         
    if uv_tensor.size()[-1] != 2:
        uv_tensor = uv_tensor.permute(0, 2, 1)  # permute to have 2 as last dim to be padded
        assert uv_tensor.size()[-1] == 2, "Tensor size not recognized"
    uv_padded = F.pad(uv_tensor, pad=(0, 1), mode="constant", value=1)  # pad only last-dim below with value 1

    kk_1 = torch.inverse(kk)

    uv_padded = uv_padded.double()
    kk_1 = kk_1.double()

    xyz_met_norm = torch.matmul(uv_padded, kk_1.t())  # More general than torch.mm
    xyz_met = xyz_met_norm * z_met

    return xyz_met

def pixel_to_camera_pt(u, v, z, camera_intrinsic, scale = 1.3):
    fx = camera_intrinsic[0][0]
    fy = camera_intrinsic[1][1]
    cx = camera_intrinsic[0][2]
    cy = camera_intrinsic[1][2]

    X = z
    Y = - (u - cx) * z / fx * scale
    Z = - (v - cy) * z / fy * scale + 0.5
    return np.array([X, Y, Z])

def xyz_from_distance(distances, xy_centers):
    """
    From distances and normalized image coordinates (z=1), extract the real world position xyz
    distances --> tensor (m,1) or (m) or float
    xy_centers --> tensor(m,3) or (3)
    """

    if isinstance(distances, float):
        distances = torch.tensor(distances).unsqueeze(0)
    if len(distances.size()) == 1:
        distances = distances.unsqueeze(1)
    if len(xy_centers.size()) == 1:
        xy_centers = xy_centers.unsqueeze(0)

    assert xy_centers.size()[-1] == 3 and distances.size()[-1] == 1, "Size of tensor not recognized"

    return xy_centers * distances / torch.sqrt(1 + xy_centers[:, 0:1].pow(2) + xy_centers[:, 1:2].pow(2))

def back_correct_angles(yaws, xyz):
    corrections = torch.atan2(xyz[:, 0], xyz[:, 2])
    yaws = yaws + corrections.view(-1, 1)
    yaws[yaws > math.pi] -= 2 * math.pi
    yaws[yaws < -math.pi] += 2 * math.pi
    # assert torch.all(yaws < math.pi) & torch.all(yaws > - math.pi)
    return yaws

def to_cartesian(rtp, mode=None):
    """convert from spherical to cartesian"""

    if isinstance(rtp, torch.Tensor):
        if mode in ('x', 'y'):
            r = rtp[:, 2]
            t = rtp[:, 0]
            p = rtp[:, 1]
        if mode == 'x':
            x = r * torch.sin(p) * torch.cos(t)
            return x.view(-1, 1)

        if mode == 'y':
            y = r * torch.cos(p)
            return y.view(-1, 1)

        xyz = rtp.clone()
        xyz[:, 0] = rtp[:, 0] * torch.sin(rtp[:, 2]) * torch.cos(rtp[:, 1])
        xyz[:, 1] = rtp[:, 0] * torch.cos(rtp[:, 2])
        xyz[:, 2] = rtp[:, 0] * torch.sin(rtp[:, 2]) * torch.sin(rtp[:, 1])
        return xyz

    x = rtp[0] * math.sin(rtp[2]) * math.cos(rtp[1])
    y = rtp[0] * math.cos(rtp[2])
    z = rtp[0] * math.sin(rtp[2]) * math.sin(rtp[1])
    return[x, y, z]

def get_keypoints(keypoints, mode):
    """
    Extract center, shoulder or hip points of a keypoint
    Input --> list or torch/numpy tensor [(m, 3, 17) or (3, 17)]
    Output --> torch.tensor [(m, 2)]
    """
    if isinstance(keypoints, (list, np.ndarray)):
        keypoints = torch.tensor(keypoints)
    if len(keypoints.size()) == 2:  # add batch dim
        keypoints = keypoints.unsqueeze(0)
    # assert len(keypoints.size()) == 3 and keypoints.size()[1] == 3, "tensor dimensions not recognized"
    assert mode in ['center', 'bottom', 'head', 'shoulder', 'hip', 'ankle']

    kps_in = keypoints[:, 0:2, :]  # (m, 2, 17)
    if mode == 'center':
        kps_max, _ = kps_in.max(2)  # returns value, indices
        kps_min, _ = kps_in.min(2)
        kps_out = (kps_max - kps_min) / 2 + kps_min   # (m, 2) as keepdims is False

    elif mode == 'bottom':  # bottom center for kitti evaluation
        kps_max, _ = kps_in.max(2)
        kps_min, _ = kps_in.min(2)
        kps_out_x = (kps_max[:, 0:1] - kps_min[:, 0:1]) / 2 + kps_min[:, 0:1]
        kps_out_y = kps_max[:, 1:2]
        kps_out = torch.cat((kps_out_x, kps_out_y), -1)

    elif mode == 'head':
        kps_out = kps_in[:, :, 0:5].mean(2)

    elif mode == 'shoulder':
        kps_out = kps_in[:, :, 5:7].mean(2)

    elif mode == 'hip':
        kps_out = kps_in[:, :, 11:13].mean(2)

    elif mode == 'ankle':
        kps_out = kps_in[:, :, 15:17].mean(2)

    return kps_out  # (m, 2)