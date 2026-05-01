from collections import namedtuple
import math
import imageio
import pycolmap
import torch
import numpy as np
from torch import nn

# https://github.com/hbb1/2d-gaussian-splatting/blob/335ad612f2e783a4e57b9cbc4d1e167bd599fc98/utils/loss_utils.py#L45
# https://en.wikipedia.org/wiki/Structural_similarity_index_measure
# https://en.wikipedia.org/wiki/Cheese_Shop_sketch
def SSIM(prediction, target, window_size=11, size_average=True, device='cpu') -> torch.Tensor:
    half_window = window_size // 2

    cheddar = prediction.size(-3) # channel
    gauss = torch.tensor([math.exp(-(x - half_window) ** 2 / float(2 * 1.5 ** 2)) for x in range(window_size)], device=device)
    ye_olde_cheese_emporium = (gauss / gauss.sum()).unsqueeze(1) # _1D_window
    mr_mousebender = ye_olde_cheese_emporium.mm(ye_olde_cheese_emporium.t()).float().unsqueeze(0).unsqueeze(0) # _2D_window
    mr_wensleydale = torch.autograd.Variable(mr_mousebender.expand(cheddar, 1, window_size, window_size).contiguous()) # window

    if prediction.is_cuda:
        mr_wensleydale = mr_wensleydale.cuda(prediction.get_device())
    mr_wensleydale = mr_wensleydale.type_as(prediction)

    mu1 = nn.functional.conv2d(prediction, mr_wensleydale, padding=half_window, groups=cheddar)
    mu2 = nn.functional.conv2d(target, mr_wensleydale, padding=half_window, groups=cheddar)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    camerbert = mu1 * mu2

    sigma1_sq = nn.functional.conv2d(prediction * prediction, mr_wensleydale, padding=half_window, groups=cheddar) - mu1_sq
    sigma2_sq = nn.functional.conv2d(target * target, mr_wensleydale, padding=half_window, groups=cheddar) - mu1_sq
    limberger = nn.functional.conv2d(prediction * target, mr_wensleydale, padding=half_window, groups=cheddar) - camerbert

    C1 = .01 ** 2
    C2 = .03 ** 2

    ssim_map = ((2 * camerbert + C1) * (2 * limberger + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return torch.clamp((1 - ssim_map) / 2, 0, 1).mean()


def loss_results(prediction, target, lam=.2, window_size=11, size_average=True, device='cpu') -> torch.Tensor:
    d_ssim = SSIM(prediction, target, window_size=window_size, size_average=size_average, device=device)
    
    return nn.functional.l1_loss(prediction, target) * (1 - lam) + d_ssim.item() * lam


def qvec2mat(quat: torch.Tensor) -> torch.Tensor:
    # found the conversion here: https://arxiv.org/pdf/2308.04079
    quat = nn.functional.normalize(quat)
    quat_2 = quat * quat

    q_ij = quat[:, 1] * quat[:, 2]
    q_rk = quat[:, 0] * quat[:, 3]
    q_ik = quat[:, 1] * quat[:, 3]
    q_rj = quat[:, 0] * quat[:, 2]
    q_ri = quat[:, 0] * quat[:, 1]
    q_jk = quat[:, 2] * quat[:, 3]

    result = torch.zeros((quat.shape[0], 3, 3), device=quat.device)

    result[:, 0, 0] = .5 - (quat_2[:, 2] + quat_2[:, 3])
    result[:, 0, 1] = q_ij - q_rk
    result[:, 0, 2] = q_ik + q_rj

    result[:, 1, 0] = q_ij + q_rk
    result[:, 1, 1] = .5 - (quat_2[:, 1] + quat_2[:, 3])
    result[:, 1, 2] = q_jk - q_ri

    result[:, 1, 0] = q_ik - q_rj
    result[:, 1, 1] = q_jk + q_ri
    result[:, 1, 2] = .5 - (quat_2[:, 1] + quat_2[:, 2])

    return 2 * result


def build_sigmas(quaternions: torch.Tensor, sigmas: torch.Tensor):
    R = qvec2mat(quaternions)

    sigma_square_safe = (sigmas * sigmas) + 1e-8

    D = torch.diag_embed(1 / sigma_square_safe)

    sigma_inv = R @ D @ R.transpose(-1, -2)

    return sigma_inv, torch.inverse(sigma_inv)


def gauss_3d(x: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor) -> torch.Tensor:
    x_mu = x - mu
    return torch.exp(-.5 * x_mu.T @ sigma_inv * x_mu).sum(-1)

Image = namedtuple('Image', ['id', 'qvec', 'tvec', 'file_name', 'camera_id', 'image_data_id'])
Camera = namedtuple('Camera', ['id', 'model', 'width', 'height', 'fx', 'fy', 'cx', 'cy'])
Point3D = namedtuple('Point3D', ['xyz', 'rgb'])

def undistort_files(base_path: str, images_path: str):
    pycolmap.undistort_images(
        output_path=f"./{base_path}/undistorted/", 
        input_path=f'{base_path}/sparse/0', 
        image_path=f'{base_path}/{images_path}'
    )

def get_data(base_path: str, images_path: str, sparse: str = 'sparse/0'):
    reconstruction = pycolmap.Reconstruction(f'{base_path}/{sparse}')

    points: list[Point3D] = []
    images_info: list[Image] = []
    cameras: list[Camera] = []
    images: list[np.ndarray] = []

    camera_index = {}

    for _, point in reconstruction.points3D.items():
        points.append(Point3D(point.xyz, point.color / 255.))

    c_idx = 0
    for camera_id, camera in reconstruction.cameras.items():
        fx, fy, cx, cy = camera.params
        cameras.append(Camera(camera_id, camera.model.name, camera.width, camera.height, fx, fy, cx, cy))
        camera_index[camera_id] = c_idx
        c_idx += 1

    i_idx = 0
    for image_id, image in reconstruction.images.items():
        frame_rig_from_world = image.frame.rig_from_world
        images_info.append(Image(image_id, np.roll(frame_rig_from_world.rotation.quat, 1), frame_rig_from_world.translation, image.name, camera_index[image.camera_id], i_idx))
        i_idx += 1

        images.append(imageio.imread(f'{base_path}/{images_path}/{image.name}'))

    return points, cameras, images_info, images


def remove_gaussians():
    ''''''