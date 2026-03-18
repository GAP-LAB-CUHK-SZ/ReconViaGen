import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
# Dictionary utils
def _dict_merge(dicta, dictb, prefix=''):
    """
    Merge two dictionaries.
    """
    assert isinstance(dicta, dict), 'input must be a dictionary'
    assert isinstance(dictb, dict), 'input must be a dictionary'
    dict_ = {}
    all_keys = set(dicta.keys()).union(set(dictb.keys()))
    for key in all_keys:
        if key in dicta.keys() and key in dictb.keys():
            if isinstance(dicta[key], dict) and isinstance(dictb[key], dict):
                dict_[key] = _dict_merge(dicta[key], dictb[key], prefix=f'{prefix}.{key}')
            else:
                raise ValueError(f'Duplicate key {prefix}.{key} found in both dictionaries. Types: {type(dicta[key])}, {type(dictb[key])}')
        elif key in dicta.keys():
            dict_[key] = dicta[key]
        else:
            dict_[key] = dictb[key]
    return dict_


def dict_merge(dicta, dictb):
    """
    Merge two dictionaries.
    """
    return _dict_merge(dicta, dictb, prefix='')


def dict_foreach(dic, func, special_func={}):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            dic[key] = dict_foreach(dic[key], func)
        else:
            if key in special_func.keys():
                dic[key] = special_func[key](dic[key])
            else:
                dic[key] = func(dic[key])
    return dic


def dict_reduce(dicts, func, special_func={}):
    """
    Reduce a list of dictionaries. Leaf values must be scalars.
    """
    assert isinstance(dicts, list), 'input must be a list of dictionaries'
    assert all([isinstance(d, dict) for d in dicts]), 'input must be a list of dictionaries'
    assert len(dicts) > 0, 'input must be a non-empty list of dictionaries'
    all_keys = set([key for dict_ in dicts for key in dict_.keys()])
    reduced_dict = {}
    for key in all_keys:
        vlist = [dict_[key] for dict_ in dicts if key in dict_.keys()]
        if isinstance(vlist[0], dict):
            reduced_dict[key] = dict_reduce(vlist, func, special_func)
        else:
            if key in special_func.keys():
                reduced_dict[key] = special_func[key](vlist)
            else:
                reduced_dict[key] = func(vlist)
    return reduced_dict


def dict_any(dic, func):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            if dict_any(dic[key], func):
                return True
        else:
            if func(dic[key]):
                return True
    return False


def dict_all(dic, func):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    for key in dic.keys():
        if isinstance(dic[key], dict):
            if not dict_all(dic[key], func):
                return False
        else:
            if not func(dic[key]):
                return False
    return True


def dict_flatten(dic, sep='.'):
    """
    Flatten a nested dictionary into a dictionary with no nested dictionaries.
    """
    assert isinstance(dic, dict), 'input must be a dictionary'
    flat_dict = {}
    for key in dic.keys():
        if isinstance(dic[key], dict):
            sub_dict = dict_flatten(dic[key], sep=sep)
            for sub_key in sub_dict.keys():
                flat_dict[str(key) + sep + str(sub_key)] = sub_dict[sub_key]
        else:
            flat_dict[key] = dic[key]
    return flat_dict


def make_grid(images, nrow=None, ncol=None, aspect_ratio=None):
    num_images = len(images)
    if nrow is None and ncol is None:
        if aspect_ratio is not None:
            nrow = int(np.round(np.sqrt(num_images / aspect_ratio)))
        else:
            nrow = int(np.sqrt(num_images))
        ncol = (num_images + nrow - 1) // nrow
    elif nrow is None and ncol is not None:
        nrow = (num_images + ncol - 1) // ncol
    elif nrow is not None and ncol is None:
        ncol = (num_images + nrow - 1) // nrow
    else:
        assert nrow * ncol >= num_images, 'nrow * ncol must be greater than or equal to the number of images'
        
    grid = np.zeros((nrow * images[0].shape[0], ncol * images[0].shape[1], images[0].shape[2]), dtype=images[0].dtype)
    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        grid[row * img.shape[0]:(row + 1) * img.shape[0], col * img.shape[1]:(col + 1) * img.shape[1]] = img
    return grid


def notes_on_image(img, notes=None):
    img = np.pad(img, ((0, 32), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if notes is not None:
        img = cv2.putText(img, notes, (0, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image_with_notes(img, path, notes=None):
    """
    Save an image with notes.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1, 2, 0)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = notes_on_image(img, notes)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# debug utils

def atol(x, y):
    """
    Absolute tolerance.
    """
    return torch.abs(x - y)


def rtol(x, y):
    """
    Relative tolerance.
    """
    return torch.abs(x - y) / torch.clamp_min(torch.maximum(torch.abs(x), torch.abs(y)), 1e-12)


# print utils
def indent(s, n=4):
    """
    Indent a string.
    """
    lines = s.split('\n')
    for i in range(1, len(lines)):
        lines[i] = ' ' * n + lines[i]
    return '\n'.join(lines)

def rotation2quad(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix).cuda()

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def quad2rotation(q):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    # bs = quad.shape[0]
    # qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    # two_s = 2.0 / (quad * quad).sum(-1)
    # rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    # rot_mat[:, 0, 0] = 1 - two_s * (qj**2 + qk**2)
    # rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    # rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    # rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    # rot_mat[:, 1, 1] = 1 - two_s * (qi**2 + qk**2)
    # rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    # rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    # rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    # rot_mat[:, 2, 2] = 1 - two_s * (qi**2 + qj**2)
    # return rot_mat
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q).cuda()

    norm = torch.sqrt(
        q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    )
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3)).to(q)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def perform_rodrigues_transformation(rvec):
    try:
        R, _ = cv2.Rodrigues(rvec)
        return R
    except cv2.error as e:
        return False

def euler2rot(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_matrix()
    return rotation_matrix

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
