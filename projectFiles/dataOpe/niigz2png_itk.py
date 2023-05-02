import os
import SimpleITK as sitk


def niigzDistribute(fileroot, saveroot):
    # fileroot = './Heart/imagesTr/'
    filename = os.listdir(fileroot)
    # saveroot = './Heart/save'
    # 读取图像
    for file in filename:
        filepath = os.path.join(fileroot, file)
        ct = sitk.ReadImage(filepath)
        ct_array = sitk.GetArrayFromImage(ct)

        (x, y, z) = ct_array.shape
        # 保存图像
        for k in range(y):
            savedImg = sitk.GetImageFromArray(ct_array[:, k, :])
            saved_name = os.path.join(saveroot, file).replace(".nii.gz", "_{:0>4d}.nii.gz".format(k))
            sitk.WriteImage(savedImg, saved_name)
