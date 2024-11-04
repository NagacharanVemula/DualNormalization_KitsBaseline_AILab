import nibabel as nib
import numpy as np
import os
import SimpleITK as sitk
from bezier_curve import bezier_curve
from tqdm import tqdm
import pickle as pkl

modality = "KiTS"

with open('/mnt/storage/ramon_data_curations/tutorials/segmentation_tutorial/kits23_dset.pkl','rb') as f: 
    tr, val, ts = pkl.load(f)


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

def save_img(slice, label, dir):
    np.savez_compressed(dir, image=slice, label=label)


def norm(vol_s):
   vol_s[vol_s>=300]=300
   vol_s[vol_s<=-100]=-100
   max = np.max(vol_s)
   min = np.min(vol_s)
   slices = 2 * (vol_s - min) / (max - min) - 1
   return slices

def nonlinear_transformation(x, p0 = [-1,-1],p3=[1,1],p1=[0.25,0.25],p2=[0.75,0.75],times_mod=100000):
   points = [p0, p1, p2, p3]
   xvals, yvals = bezier_curve(points, nTimes=times_mod)
   xvals, yvals = np.sort(xvals), yvals
   nonlinear_x = np.interp(x, xvals, yvals)
   return nonlinear_x



def save_test_npz(data_root, modality, target_root):
    # list_dir = os.listdir(data_root)
    dset_2 = data_root
    tbar = tqdm(dset_2, ncols=70)
    count = 0
    # count_sample = 0
    for name in tbar:
        nib_img = nib.load(name["image"])
        nib_mask = nib.load(name["label"])

        affine = nib_img.affine.copy()
        
        slices = nib_img.get_fdata()
        masks = nib_mask.get_fdata()
        masks =  np.where((masks == 0) | (masks == 1), masks, 0)

        

        if not os.path.exists(os.path.join(target_root, modality)):
            os.makedirs(os.path.join(target_root, modality))
        
        for i in range(slices.shape[0]):
            slice = norm(slices[i,:,:])
            mask = masks[i,:,:]
            save_img(slice, mask, os.path.join(target_root, modality, 'val_sample{}.npz'.format(count)))
            count += 1
        
        # count_sample+=1
        # if count_sample==2:
        #     break





def main(data_root, modality, target_root):
    # list_dir = os.listdir(data_root)
    dset_1 = data_root
    tbar = tqdm(dset_1, ncols=70)
    count = 0
    # count_sample = 0
    for name in tbar:
        nib_img = nib.load(name["image"])
        nib_mask = nib.load(name["label"])
        
        affine = nib_img.affine.copy()
        
        slices = nib_img.get_fdata()
        masks = nib_mask.get_fdata()
        masks =  np.where((masks == 0) | (masks == 1), masks, 0)
        # masks[masks != 0] = 1

        # slices = norm(slices)

        # slices, nonlinear_slices_1 = nonlinear_transformation(slices)

        if not os.path.exists(os.path.join(target_root, modality + '_ss')):
            os.makedirs(os.path.join(target_root, modality + '_ss'))
        if not os.path.exists(os.path.join(target_root, modality + '_sd')):
            os.makedirs(os.path.join(target_root, modality + '_sd'))
        
        v1 = 0.25
        v2 = 0.75

        for i in range(slices.shape[0]):
            slice = norm(slices[i,:,:])
            mask = masks[i,:,:]
            kidney_mask = mask ==1

            """
            Source-Similar
            """
            mod_kidney = nonlinear_transformation(slice[kidney_mask],p0 = [-1,-1,],p1=[-1,-1],p2=[1,1],p3 =[1,1])
            new_kidney_1 = np.copy(slice)
            new_kidney_1[kidney_mask]= mod_kidney

            mod_kidney = nonlinear_transformation(slice[kidney_mask],p0 = [-1,-1,],p1=[-v1,v1],p2=[v1,-v1],p3 = [1,1])
            new_kidney_2 = np.copy(slice)
            new_kidney_2[kidney_mask]= mod_kidney

            mod_kidney = nonlinear_transformation(slice[kidney_mask],p0 = [-1,-1,],p1=[-v2,v2],p2=[v2,-v2],p3 = [1,1])
            new_kidney_3 = np.copy(slice)
            new_kidney_3[kidney_mask]= mod_kidney

            """
            Source-Dissimilar
            """
            mod_kidney = nonlinear_transformation(slice[kidney_mask],p0 = [1,1,],p1=[1,1],p2=[-1,-1],p3 = [-1,-1])
            new_kidney_4 = np.copy(slice)
            new_kidney_4[kidney_mask]= mod_kidney

            mod_kidney = nonlinear_transformation(slice[kidney_mask],p0 = [1,1,],p1=[v1,-v1],p2=[-v1,v1],p3 = [-1,-1])
            new_kidney_5 = np.copy(slice)
            new_kidney_5[kidney_mask]= mod_kidney

            mod_kidney = nonlinear_transformation(slice[kidney_mask],p0 = [1,1,],p1=[v2,-v2],p2=[-v2,v2],p3 = [-1,-1])
            new_kidney_6 = np.copy(slice)
            new_kidney_6[kidney_mask]= mod_kidney



            save_img(new_kidney_1, mask, os.path.join(target_root, modality + '_ss', 'sample{}.npz'.format(count)))
            save_img(new_kidney_4, mask, os.path.join(target_root, modality + '_sd', 'sample{}.npz'.format(count)))
            count += 1

            save_img(new_kidney_2, mask, os.path.join(target_root, modality + '_ss', 'sample{}.npz'.format(count)))
            save_img(new_kidney_5, mask, os.path.join(target_root, modality + '_sd', 'sample{}.npz'.format(count)))
            count += 1

            save_img(new_kidney_3, mask, os.path.join(target_root, modality + '_ss', 'sample{}.npz'.format(count)))
            save_img(new_kidney_6, mask, os.path.join(target_root, modality + '_sd', 'sample{}.npz'.format(count)))
            count += 1

        # count_sample+=1
        # if count_sample==2:
        #     break



if __name__ == '__main__':
    print("Processing started!")
    tr[64]["image"] = '/mnt/storage/charan/kits_resized/case_00425/imaging_resized.nii.gz'
    tr[64]["label"] = '/mnt/storage/charan/kits_resized/case_00425/segmentation_resized.nii.gz'

    tr[271]["image"] = '/mnt/storage/charan/kits_resized/case_00160/imaging_resized.nii.gz'
    tr[271]["label"] = '/mnt/storage/charan/kits_resized/case_00160/segmentation_resized.nii.gz'

    data_root = tr
    target_root =  "/mnt/storage/charan/npz_data/train/"
    modality = 'KiTS'
    main(data_root, modality, target_root)
    print("It's done")

    data_root = val
    target_root = "/mnt/storage/charan/npz_data1/val/"
    modality_list = ['KiTS']
    for modality in modality_list:
        save_test_npz(data_root, modality, target_root)
    print("It's done")