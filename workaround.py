# import nibabel
# import numpy as np
# import os
# from PIL import Image
# from skimage import io

# def save_slice_as_grayscale(slice_data, output_dir, name):
#     # Normalize the slice data to [0, 255]
#     normalized_slice = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
#     # Save the grayscale slice as a PNG
#     output_path = os.path.join(output_dir, f"{name}.jpg")
#     io.imsave(output_path, normalized_slice)


# save_dir = "BTCV"
# training_image_root = "RawData/Training/img"
# training_label_root = "RawData/Training/label"
# training_image_paths = sorted([os.path.join(training_image_root, f) for f in os.listdir(training_image_root)])
# training_label_paths = sorted([os.path.join(training_label_root, f) for f in os.listdir(training_label_root)])
# mode = ["training", "validation"]
# print(training_image_paths, training_label_paths)
# image_dir = os.path.join(save_dir, "images")
# for index, file in enumerate(training_image_paths):
#     if index >= 25:
#         output_dir = os.path.join(image_dir, "validation")
#     else:
#         output_dir = os.path.join(image_dir, "training")
#     img  = nibabel.load(file)
#     img = img.get_fdata()
#     img_name = os.path.basename(file)
#     img_name = img_name.split(".")[0][3:]
#     # print(img.shape)
#     print(img_name)
#     _, _, num_slices = img.shape
#     for slice in range(num_slices):
#         slice_data = img[:,:,slice]
#         name = f"{img_name}_{str(slice).zfill(3)}"
#         save_slice_as_grayscale(slice_data, output_dir, name)
#         print(f"Saving {name} to {output_dir}.")
        
        
# def save_slice_as_grayscale(slice_data, output_dir, name):
#     normalized_slice = slice_data.astype(np.uint8)
#     # Save the grayscale slice as a PNG
#     output_path = os.path.join(output_dir, f"{name}.png")
#     io.imsave(output_path, normalized_slice)


# label_dir = os.path.join(save_dir, "annotations")
# for index, file in enumerate(training_image_paths):
#     if index >= 25:
#         output_dir = os.path.join(label_dir, "validation")
#     else:
#         output_dir = os.path.join(label_dir, "training")
#     img  = nibabel.load(file)
#     img = img.get_fdata()
#     img_name = os.path.basename(file)
#     img_name = img_name.split(".")[0][5:]
#     # print(img.shape)
#     print(img_name)
#     _, _, num_slices = img.shape
#     for slice in range(num_slices):
#         slice_data = img[:,:,slice]
#         name = f"{img_name}_{str(slice).zfill(3)}"
#         save_slice_as_grayscale(slice_data, output_dir, name)
#         print(f"Saving {name} to {output_dir}.")
import torch
import torch.nn as nn
import numpy as np
classes = [100, 10, 10, 10,10]
qdims = np.cumsum([0, 100]+[10]*4)

# print(qdims)
prompt_no_obj_embed = nn.ModuleList([nn.Linear(256, 1) for _ in classes[1:]])
class_embed = nn.Linear(256, 150 + 1)
prompt_feat = nn.ModuleList([nn.Embedding(10, 256) for _ in classes[1:]])
for n in range(1, len(classes)-1):
    # print(n)
    print(prompt_feat[-1].weight.unsqueeze(1).shape)
    # prompt = prompt_feat[-1].weight.unsqueeze(1).repeat(1,2,1).transpose(0,1)
    # print(prompt.shape)
    # outputs_classes = class_embed(prompt)
    # no_obj_logit = prompt_no_obj_embed[n-1](prompt)
    # print(no_obj_logit.shape)
    # # outputs_classes[:, qdims[n]:qdims[n+1], -1] = no_obj_logit[:, :, 0]
    # print(outputs_classes.shape)