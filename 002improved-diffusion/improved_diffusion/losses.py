"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    标准正态累积分布函数的快速近似。

    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))

#累计函数的差分 模拟 离散分布
def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).

    高斯分布概率密度函数（PDF）和累积分布函数(CDF)
    x是由[0...255]rescaled到[-1,1]的，也就是说相邻的像素值的差值大小为2/255.0,也就是一个单位大小为2/255.0；
    已知x所属高斯分布函数的均值和方差，由于像素值是离散的值，所以需要获得离散的高斯分布函数;
    定义：离散的p(x)等于连续高斯分布概率密度函数中[x-1/255，x+1/255]所占的面积(即以x为中心一个单位内的面积)
    要求这个值就需要获得其累积分布函数，由其累积分布函数的差分得到，离散的p(x)=CDF(x+半个单位)-CDF(x-半个单位)；
    由于高斯分布概率密度函数是不可积的，无原函数，因此只能通过近似得到，并且通常是近似标准的高斯分布的累积分布函数
    因此需要将已知的高斯分布转换到标准的高斯分布，再通过标准的高斯分布的累积分布函数的差分获得离散的p(x)；
    当x < -0.999时，考虑到这个位置概率较小，所以其离散的值不是以x为中心一个单位内的面积，而是x+半个单位左边的全部面积，即log_cdf_plus
    当x>0.9999时，考虑到这个位置概率较小，所以其离散的值不是以x为中心一个单位内的面积，而是x-半个单位右边的全部面积，即1-cdf_min(log_log_one_minus_cdf_min);


    """



    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))

    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))

    #区间的值
    cdf_delta = cdf_plus - cdf_min

    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
