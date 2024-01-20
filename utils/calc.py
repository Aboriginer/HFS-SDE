import torch
from torchmetrics import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    MeanSquaredError,
)
import scipy.io as scio
import numpy as np
import cv2
import scipy.io as scio


def process_label_tensor(label_tensor):
    # Convert the label tensor to a NumPy array
    label = label_tensor.cpu().numpy()  # Assuming label_tensor is on CPU

    # Threshold for binary mask
    threshold = 0.03 * np.max(np.abs(label))
    mask1 = np.abs(label) > threshold

    # Convert the mask to a binary image (0 or 255)
    mask1 = np.uint8(mask1) * 255

    # Create a circular structuring element with a radius of 10
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    # Erode the mask using the structuring element
    mask1 = cv2.erode(mask1, se)

    # Dilate the mask using a larger circular structuring element
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    mask1 = cv2.dilate(mask1, se)

    # Convert the NumPy array back to a PyTorch tensor
    mask1_tensor = torch.from_numpy(mask1).to(label_tensor.device)

    return mask1_tensor


def Evaluation_metrics(label, recon, mask=False):
    if mask:
        mask = scio.loadmat("mask/photom_mask.mat")["mask"]
        mask = mask.astype(np.complex128)
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask).to(label)
        print(mask.shape)
        print(recon.shape)
        recon = torch.mul(mask, recon)

        print("make photom")
        pass

    if torch.is_tensor(label):
        pass
    else:
        label = torch.from_numpy(label)
        recon = torch.from_numpy(recon)
    label = torch.abs(label).type(torch.FloatTensor)
    recon = torch.abs(recon).type(torch.FloatTensor)
    label = torch.unsqueeze(label, 0)
    label = torch.unsqueeze(label, 0)
    recon = torch.unsqueeze(recon, 0)
    recon = torch.unsqueeze(recon, 0)

    if len(recon.size()) != 4:
        print(recon.size())
        recon = recon.view(1, 1, recon.size()[-1], recon.size()[-1])
        label = label.view(1, 1, recon.size()[-1], recon.size()[-1])

    ssim = StructuralSimilarityIndexMeasure()
    res = ssim(recon, label)

    psnr = PeakSignalNoiseRatio()
    psnrs = psnr(recon, label)

    # Assuming 'recon' and 'label' are NumPy arrays
    recon = recon / torch.max(recon)
    label = label / torch.max(label)
    err = torch.abs(recon - label)

    nmse = (
        torch.linalg.norm(err.reshape(-1)) ** 2
        / torch.linalg.norm(label.reshape(-1)) ** 2
    )

    print(res)
    print(psnrs)
    print(nmse)
    return res, psnrs, nmse
