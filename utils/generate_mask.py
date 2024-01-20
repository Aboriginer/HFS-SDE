import os
import sys
import torch
import numpy as np
import scipy.io as scio
from numpy.lib.stride_tricks import as_strided
from PIL import Image


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)


def cartesian_mask(shape, acc, sample_n):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5 / (Nx / 10.0) ** 2)
    lmda = Nx / (2.0 * acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1.0 / Nx

    if sample_n:
        pdf_x[Nx // 2 - sample_n // 2 : Nx // 2 + sample_n // 2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx // 2 - sample_n // 2 : Nx // 2 + sample_n // 2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    return mask


def get_blur_mask(image_size, length):
    mask = torch.zeros([1, 1, image_size, image_size], dtype=torch.complex128)
    x_start = int((image_size - length) / 2)
    x_end = int((image_size + length) / 2)
    mask[:, :, x_start:x_end, x_start:x_end] = 1.0

    mask_result_file = os.path.join("mask", "center_length" + str(length) + ".mat")
    mask_datadict = {"mask": np.squeeze(mask.numpy())}
    scio.savemat(mask_result_file, mask_datadict)

    return mask


def get_blur_mask(image_size, length):
    mask = torch.zeros([1, 1, image_size, image_size], dtype=torch.complex128)
    x_start = int((image_size - length) / 2)
    x_end = int((image_size + length) / 2)
    mask[:, :, x_start:x_end, x_start:x_end] = 1.0

    mask_result_file = os.path.join("mask", "center_length" + str(length) + ".mat")
    mask_datadict = {"mask": np.squeeze(mask.numpy())}
    scio.savemat(mask_result_file, mask_datadict)

    return mask


def get_uniform_random_mask(image_size, acc, acs_lines=18):
    center_line_idx = np.arange(
        (image_size - acs_lines) // 2, (image_size + acs_lines) // 2
    )
    outer_line_idx = np.setdiff1d(np.arange(image_size), center_line_idx)
    np.random.shuffle(outer_line_idx)
    print(outer_line_idx)

    lines_num = int(image_size / acc) - acs_lines
    random_line_idx = outer_line_idx[0:lines_num]
    print(random_line_idx)

    mask = np.zeros((image_size))
    mask[center_line_idx] = 1.0
    mask[random_line_idx] = 1.0

    mask = np.repeat(mask[np.newaxis, :], image_size, axis=0)

    mask_result_file = os.path.join(
        "mask", "256random_uniform_acc" + str(acc) + "_acs" + str(acs_lines) + ".mat"
    )
    mask_datadict = {"mask": np.squeeze(mask)}
    scio.savemat(mask_result_file, mask_datadict)


def get_equispaced_mask(mask_type, acc, acs_lines=16, total_lines=320):
    center_line_idx = np.arange(
        (total_lines - acs_lines) // 2, (total_lines + acs_lines) // 2
    )
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)

    random_line_idx = outer_line_idx[::acc]
    print(random_line_idx)

    mask = np.zeros((total_lines))
    mask[center_line_idx] = 1.0
    if mask_type == "low_frequency":
        mask[random_line_idx] = 0.0
    else:
        mask[random_line_idx] = 1.0

    mask = np.repeat(mask[np.newaxis, :], total_lines, axis=0)
    if mask_type == "low_frequency":
        mask_result_file = os.path.join(
            "mask", "low_frequency_acs" + str(acs_lines) + ".mat"
        )
    else:
        mask_result_file = os.path.join(
            "mask", "uniform_acc" + str(acc) + "_acs" + str(acs_lines) + ".mat"
        )
    mask_datadict = {"mask": np.squeeze(mask)}
    scio.savemat(mask_result_file, mask_datadict)

    return mask


def get_cartesian_mask(acc, acs_lines=24, image_size=320):
    shape = (1, image_size, image_size)
    mask = cartesian_mask(shape, acc, sample_n=acs_lines)
    mask = np.transpose(mask, (0, 2, 1))

    mask_result_file = os.path.join(
        "mask", "cartesian_acc" + str(acc) + "_acs" + str(acs_lines) + ".mat"
    )
    mask_datadict = {"mask": np.squeeze(mask)}
    scio.savemat(mask_result_file, mask_datadict)
    print("generate cartesian mask, acc =", acc)


def get_UIH_mask():
    mask = scio.loadmat("UIH/ks_mask_a4.mat")["ks_mask"]

    mask = np.repeat(mask[np.newaxis, :], 512, axis=0)

    mask_result_file = os.path.join("UIH", "uniform_acc_4.mat")
    mask_datadict = {"mask": np.squeeze(mask)}
    scio.savemat(mask_result_file, mask_datadict)


def tensor_to_gray_image(tensor, filename):
    # 将张量的值缩放到[0, 255]并转为byte类型
    tensor = tensor.mul(255).byte()
    tensor = tensor.cpu().numpy()

    # 使用PIL创建灰度图像
    im = Image.fromarray(tensor, "L")  # 'L' 表示灰度图

    # 保存图像
    im.save(filename)


def main():
    import random

    # 获取当前的随机种子值
    seed_value = random.getstate()[1][0]

    # 打印随机种子值
    print("当前随机种子值:", seed_value)

    get_uniform_random_mask(256, 20, 1)
    mask_file = "/data0/yuewang/code/multiscale_score_based/mask/256random_uniform_acc20_acs1.mat"
    mask = scio.loadmat(mask_file)["mask"]
    mask = mask.astype(np.complex128)
    mask = torch.from_numpy(mask)
    tensor_to_gray_image(mask, "mask.png")


if __name__ == "__main__":
    sys.exit(main())
