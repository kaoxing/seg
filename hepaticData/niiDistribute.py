import os
import SimpleITK as sitk


def dicomDistribute(fileroot, saveroot, type="nii", axial="x"):
    files = os.listdir(fileroot)
    for i, file1 in enumerate(files):
        filename = os.listdir(os.path.join(fileroot, file1))
        # filename = os.listdir(fileroot)
        # 读取图像
        for n, file in enumerate(filename):
            if file.startswith("lge") or not file.endswith(".nrrd"):
                continue

            filepath = os.path.join(fileroot, file1, file)
            print(i, filepath)
            ct = sitk.ReadImage(filepath)
            ct_array = sitk.GetArrayFromImage(ct)
            size = 0
            if axial == "x":
                size = ct_array.shape[0]
            elif axial == "y":
                size = ct_array.shape[1]
            else:
                size = ct_array.shape[2]
            # 保存图像
            for k in range(size):
                savedImg = 0
                if axial == "x":
                    savedImg = sitk.GetImageFromArray(ct_array[k, :, :])
                elif axial == "y":
                    savedImg = sitk.GetImageFromArray(ct_array[:, k, :])
                else:
                    savedImg = sitk.GetArrayFromImage(ct_array[:, :, k])

                if type == 'nii':
                    saved_name = os.path.join(saveroot, file).replace(".nii", "_{:0>3d}_{:0>4d}.nii".format(i, k))
                    # saved_name = os.path.join(saveroot, file).replace(".nii", "_{:0>3d}_{:0>4d}.nii".format(n, k))
                    sitk.WriteImage(savedImg, saved_name)
                elif type == 'nrrd':
                    saved_name = os.path.join(saveroot, file).replace(".nrrd", "_{:0>3d}_{:0>4d}.nrrd".format(i, k))
                    # saved_name = os.path.join(saveroot, file).replace(".nrrd", "_{:0>3d}_{:0>4d}.nrrd".format(n, k))
                    sitk.WriteImage(savedImg, saved_name)


if __name__ == '__main__':
    dicomDistribute("G:\\2018 Atrial Segmentation Challenge COMPLETE DATASET\\Testing Set\\",
                    "D:\\pythonProject\\seg\\mytrain\\HeartData\\test\\label", type="nrrd", axial="x")
