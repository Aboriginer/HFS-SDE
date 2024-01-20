import os
import h5py
import pickle

# Specify the directory where the H5 files are located
directory = "data/photom/kspace/"  # Please replace with your directory path
output_pkl = "data/data_slice.pkl"  # Path to the output PKL file

# Get all H5 files in the directory
h5_files = [f for f in os.listdir(directory) if f.endswith(".h5")]

# Read the old PKL file if it exists
try:
    with open(output_pkl, "rb") as pkl_file:
        old_data_dict = pickle.load(pkl_file)
except FileNotFoundError:
    old_data_dict = {}


# 创建一个字典，以存储第一个维度的大小
data_dict = {}

# 遍历每个h5文件
for h5_file in h5_files:
    full_path = os.path.join(directory, h5_file)

    # 打开h5文件
    with h5py.File(full_path, "r") as h5f:
        # kspace
        if "kspace" in h5f.keys():
            data = h5f["kspace"][:]
            # 记录第一个维度的大小
            data_dict[h5_file] = data.shape[0]
            print(h5_file)

old_data_dict.update(data_dict)
# 将字典保存为pkl文件
with open(output_pkl, "wb") as pkl_file:
    pickle.dump(old_data_dict, pkl_file)

print(f"Data has been saved to {output_pkl}")
print(len(old_data_dict.values()))
