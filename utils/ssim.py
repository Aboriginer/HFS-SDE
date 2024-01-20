import torch
from torchmetrics import StructuralSimilarityIndexMeasure
import scipy.io as scio


label = scio.loadmat("label_1.mat")["label"]
recon = scio.loadmat("recon_1.mat")["recon"]

label = torch.from_numpy(label)
recon = torch.from_numpy(recon)
label = torch.abs(label).type(torch.FloatTensor)
recon = torch.abs(recon).type(torch.FloatTensor)
label = torch.unsqueeze(label, 0)
label = torch.unsqueeze(label, 0)
recon = torch.unsqueeze(recon, 0)
recon = torch.unsqueeze(recon, 0)

ssim = StructuralSimilarityIndexMeasure()
res = ssim(recon, label)
print(res)
