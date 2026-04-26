import math
import torch
from torch import nn


# https://github.com/hbb1/2d-gaussian-splatting/blob/335ad612f2e783a4e57b9cbc4d1e167bd599fc98/utils/loss_utils.py#L45
# https://en.wikipedia.org/wiki/Structural_similarity_index_measure
# https://en.wikipedia.org/wiki/Cheese_Shop_sketch
def SSIM(prediction, target, window_size=11, size_average=True, device='cpu') -> torch.Tensor:
    half_window = window_size // 2

    cheddar = prediction.size(-3)
    gauss = torch.tensor([math.exp(-(x - half_window) ** 2 / float(2 * 1.5 ** 2)) for x in range(window_size)], device=device)
    ye_olde_cheese_emporium = (gauss / gauss.sum()).unsqueeze(1)
    mr_mousebender = ye_olde_cheese_emporium.mm(ye_olde_cheese_emporium.t()).float().unsqueeze(0).unsqueeze(0)
    mr_wensleydale = torch.autograd.Variable(mr_mousebender.expand(cheddar, 1, window_size, window_size).contiguous())

    if prediction.is_cuda:
        mr_wensleydale.cuda(prediction.get_device())
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

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


def loss_results(prediction, target, lam=.2, window_size=11, size_average=True, device='cpu') -> torch.Tensor:
    d_ssim = (1 - SSIM(prediction, target, window_size=window_size, size_average=size_average, device=device)).squeeze(0)
    
    return nn.functional.l1_loss(prediction, target) * (1 - lam) + d_ssim.item() * lam


def build_sigma_inv(quat: torch.Tensor, sigma: torch.Tensor):
    w_2 = quat[0] * quat[0]
    x_2 = quat[1] * quat[1]
    y_2 = quat[2] * quat[2]
    z_2 = quat[3] * quat[3]

    xy = quat[1] * quat[2]
    wz = quat[0] * quat[3]
    xz = quat[1] * quat[3]
    wy = quat[0] * quat[2]
    wx = quat[0] * quat[1]
    yz = quat[2] * quat[3]

    R = torch.Tensor([
        [2 * (w_2 + x_2) - 1, 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 2 * (w_2 + y_2) - 1, 2 * (xz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 2 * (w_2 + z_2) - 1],
    ], device=quat.device)

    S = torch.eye(3, device=sigma.device)
    
    sigma_square_safe = (sigma * sigma) + 1e-8

    S *= 1 / sigma_square_safe

    return R @ S @ R.T
