#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os
os.chdir('C:\\Users\\somd7w\\Desktop\\DL_Projects\\preproc_cntr')
os.getcwd()


# In[23]:


import src.unet3d.utils.utils
import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from nilearn.image import resample_img, crop_img,index_img, resample_to_img,resample_to_img, reorder_img, new_img_like
from niwidgets import NiftiWidget


# In[24]:


# Function to calculate voxel co-ordinate in mm space
def get_coord(img,ijk):
    M = img.affine[:3,:3]
    T = img.affine[:3,3]
    return M.dot(ijk) + T

def get_center_voxel(img):
    img_data = img.get_data()
    return (np.array(img_data.shape) - 1) / 2.


# In[25]:


patname = "Elast-463"


# In[26]:


path_t2 = os.path.abspath("preprocessed/"+patname+"/T2.nii")


# In[27]:


path_mask = os.path.abspath("preprocessed/"+patname+"/Label.nii")


# In[28]:


T2 = nib.load(path_t2)
MASK = nib.load(path_mask)
T2_org = nib.load(path_t2)
#Reorder to RAS
T2 = reorder_img(T2, resample='continuous')
MASK = reorder_img(MASK, resample='nearest')
image = T2


# In[29]:


print(T2.affine)
print(nib.aff2axcodes(T2.affine))


# In[30]:


new_shape = (256,256,64)
zoom_level = np.divide(new_shape, image.shape)
current_spacing = [1,1,1]*T2.affine[:3,:3]
new_spacing = np.divide(current_spacing, zoom_level)


# In[31]:


new_affine =  np.eye(4)
new_affine[:3, :3] =    np.eye(3)*new_spacing
new_affine[:3,3] = T2.affine[:3,3]


# In[32]:


T2_new = resample_img(image,target_shape=new_shape,target_affine=new_affine,interpolation="continuous")
MASK_new = resample_img(MASK,target_shape=new_shape,target_affine=new_affine,interpolation="nearest")


# In[33]:


# test_widget = NiftiWidget(T2_org)
# test_widget.nifti_plotter()


# In[34]:


# print(T2_org.affine)
# print(T2_org.shape)


# In[35]:


# test_widget = NiftiWidget(T2_new)
# test_widget.nifti_plotter()


# In[36]:


print(T2_new.affine)
print(T2_new.shape)


# In[37]:


# test_widget = NiftiWidget(MASK_new)
# test_widget.nifti_plotter()


# In[38]:


from src.unet3d.utils.utils import get_imstats
from src.unet3d.normalize import get_volume
#print("T2_stats = ",get_imstats(T2))
#print("T2_new_stats = ",get_imstats(T2_new))

print("old_voxel_vol = %.2f" %  np.prod(np.asarray(get_imstats(T2)[3:])))

print("new_voxel_vol = %.2f" %  np.prod(np.asarray(get_imstats(T2_new)[3:])))

print("T2_vol = %.2f" % get_volume(get_imstats(T2),units="cm"))
print("T2_new_vol = %.2f" % get_volume(get_imstats(T2_new),units="cm"))

old_vol = get_volume(get_imstats(T2),mask=MASK.get_data(),label=1,units="cm")
new_vol = get_volume(get_imstats(T2_new),mask=MASK_new.get_data(),label=1,units="cm")

print("T2_liver_vol = %.2f" % old_vol)
print("T2_new_liver_vol = %.2f" % new_vol)

rel_err = (old_vol-new_vol)*100/new_vol

print("liver_vol_err percent = %.2f" % rel_err)

print("new image shape", T2_new.shape)


# In[39]:


T2_new.header['xyzt_units']=2


# In[40]:


T2_new.to_filename('jupyter_out/'+ patname + '_new.nii.gz')


# In[ ]:




