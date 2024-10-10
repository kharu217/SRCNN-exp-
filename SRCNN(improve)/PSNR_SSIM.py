#%%
import matplotlib.figure
import matplotlib.pyplot
from model import SRCNN
import matplotlib
import torch
import skimage
import cv2
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_f = SRCNN()
model_f.load_state_dict(torch.load(r'SRCNN(improve)\models\face_m.h5', map_location=device))

model_n = SRCNN()
model_n.load_state_dict(torch.load(r'SRCNN(improve)\models\normal.h5', map_location=device))

test_img = "testing.jpg"

temp_img_r = cv2.imread(test_img)
cvtemp = cv2.cvtColor(temp_img_r, cv2.COLOR_BGR2RGB)
input_img_r = cvtemp.astype(np.float32)/255.0
input_img_r = input_img_r.transpose((2, 0, 1))
input_img_r = torch.tensor(input_img_r).unsqueeze(0)

with torch.no_grad() :
    f_model_img = model_f(input_img_r)
    n_model_img = model_n(input_img_r)

f_model_img = f_model_img.squeeze().cpu().numpy().transpose((1,2,0))
n_model_img = n_model_img.squeeze().cpu().numpy().transpose((1,2,0))
compare_img = cv2.imread(test_img)
#temp_img_r = temp_img_r.squeeze()

print(type(compare_img))
print(type(f_model_img))
print(type(n_model_img))

fig = matplotlib.pyplot.figure()

print(f_model_img.shape)
print(n_model_img.shape)
print(compare_img.shape)

psnr_f_o = skimage.metrics.peak_signal_noise_ratio(compare_img, f_model_img)
ssim_f_o = skimage.metrics.structural_similarity(compare_img, f_model_img, channel_axis=2)

psnr_n_o = skimage.metrics.peak_signal_noise_ratio(compare_img, n_model_img)
ssim_n_o = skimage.metrics.structural_similarity(compare_img, n_model_img, channel_axis=2)

psnr_o = skimage.metrics.peak_signal_noise_ratio(compare_img, compare_img)
ssim_o = skimage.metrics.structural_similarity(compare_img, compare_img, channel_axis=2)

ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(f_model_img)
ax1.set_xlabel(f'psnr : {psnr_f_o}\n ssim : {ssim_f_o}')

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(n_model_img)
ax2.set_xlabel(f'psnr : {psnr_n_o}\n ssim : {ssim_n_o}')

ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(compare_img)
ax2.set_xlabel(f'psnr : {psnr_o}\n ssim : {ssim_o}')

matplotlib.pyplot.show()
# %%