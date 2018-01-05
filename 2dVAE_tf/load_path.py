##this file is a copy. There is another one in VAE/notebook_mri


import os
#I use a list in terms of [dataset_code, 2d/3d, data_type ]

class ImagePath():
    def __init__(self, _dataset, _3d, _info):
        self._dataset=_dataset
        self._3d=_3d
        self._info=_info
    def __hash__(self):
        return hash((self._dataset, self._3d, self._info))

    def __eq__(self, other):
        return (self._dataset, self._3d, self._info) == (other._dataset.upper(), other._3d, other._info)

class LabelPath():
    def __init__(self, _dataset, _onehot, _all_fields, _info=""):
        self._dataset = _dataset
        self._onehot = _onehot
        self._all_fields = _all_fields
        self._info=_info
    def __hash__(self):
        return hash((self._dataset, self._onehot, self._all_fields, self._info))

    def __eq__(self, other):
        return (self._dataset, self._onehot, self._all_fields, self._info) == (other._dataset.upper(), other._onehot, other._all_fields, other._info)


current_path = os.getcwd()

if ("dilettamilana" in current_path):
    pass
else:
    home = "F:\\Diletta\\tesi_dataset\\"
    mac=False

#for labels
#_dataset, _onehot_all_fields, _info
images_dict={ImagePath(_dataset="HARP", _3d=False, _info="segmented"): "Harp\\images_hippo_volumes.npy",
             ImagePath(_dataset="HARP", _3d=True, _info="norm"): "Harp\\images_3d_norm.npy",
             ImagePath(_dataset="HARP", _3d=True, _info="orig"): "Harp\\images_3d_orig.npy",
             ImagePath(_dataset="HARP", _3d=True,_info="orig_patch26_32x32x64"): "Harp\\orig_getpatch26_correct_32x32x64.npy",
             ImagePath(_dataset="HARP", _3d=True, _info="orig_patch26"): "Harp\\orig_getpatch26_correct_32x64x32.npy",
             ImagePath(_dataset="HARP", _3d=True, _info="orig_patch6"): "Harp\\orig_getpatch6_correct_64x64x128.npy",
             ImagePath(_dataset="HARP", _3d=True, _info="orig_patch16"): "Harp\\orig_getpatch16_correct_64x64x32.npy",
             ImagePath(_dataset="HARP", _3d=True, _info="orig_patch17"): "Harp\\orig_getpatch17_correct_64x64x64.npy",
             ImagePath(_dataset="HARP", _3d=True, _info="orig_patch19"): "Harp\\orig_getpatch19_correct_64x64x128.npy",
             ImagePath(_dataset="HARP", _3d=True, _info="orig_patch19_64x64x80"): "Harp\\orig_getpatch19_correct_64x64x80.npy",
             ImagePath(_dataset="HARP", _3d=True, _info="orig_patch27"): "Harp\\orig_getpatch27_correct_64x64x64.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_coronal_64centerslices"): "Harp\\orig_coronal_64centerslices.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_coronal_32centerslices"): "Harp\\orig_coronal_32centerslices.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_coronal_4centerslices"): "Harp\\orig_coronal_4centerslices.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_axial_64centerslices"): "Harp\\orig_axial_64centerslices.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_axial_32centerslices"): "Harp\\orig_axial_32centerslices.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_axial_4centerslices"): "Harp\\orig_axial_4centerslices.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_sagittal_64centerslices"): "Harp\\orig_sagittal_64centerslices.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_sagittal_32centerslices"): "Harp\\orig_sagittal_32centerslices.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_sagittal_4centerslices"): "Harp\\orig_sagittal_4centerslices.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_coronal_64centerslices_flattened"): "Harp\\orig_coronal_64centerslices_flattened.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_coronal_32centerslices_flattened"): "Harp\\orig_coronal_32centerslices_flattened.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_coronal_4centerslices_flattened"): "Harp\\orig_coronal_4centerslices_flattened.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_axial_64centerslices_flattened"): "Harp\\orig_axial_64centerslices_flattened.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_axial_32centerslices_flattened"): "Harp\\orig_axial_32centerslices_flattened.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_axial_4centerslices_flattened"): "Harp\\orig_axial_4centerslices_flattened.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_sagittal_64centerslices_flattened"): "Harp\\orig_sagittal_64centerslices_flattened.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_sagittal_32centerslices_flattened"): "Harp\\orig_sagittal_32centerslices_flattened.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="orig_sagittal_4centerslices_flattened"): "Harp\\orig_sagittal_4centerslices_flattened.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="norm_axial_64centerslices"): "Harp\\norm_axial_64centerslices.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="norm_axial_32centerslices"): "Harp\\norm_axial_32centerslices.npy",
             ImagePath(_dataset="HARP", _3d=False, _info="norm_axial_4centerslices"): "Harp\\norm_axial_4centerslices.npy",

             ImagePath(_dataset="AIBL",_3d=False, _info="segmented"):"AIBL\\images_hippo_volumes.npy",
             ImagePath(_dataset="AIBL",_3d=True, _info="norm"):"AIBL\\images_3d_norm.npy",
             ImagePath(_dataset="AIBL",_3d=True, _info="orig"):"AIBL\\images_3d_orig.npy",
             ImagePath(_dataset="AIBL", _3d=True,_info="orig_patch26_32x32x64"): "AIBL\\orig_getpatch26_correct_32x32x64.npy",
             ImagePath(_dataset="AIBL", _3d=True, _info="orig_patch26"): "AIBL\\orig_getpatch26_correct_32x64x32.npy",
             ImagePath(_dataset="AIBL", _3d=True, _info="orig_patch6"): "AIBL\\orig_getpatch6_correct_64x64x128.npy",
             ImagePath(_dataset="AIBL", _3d=True, _info="orig_patch16"): "AIBL\\orig_getpatch16_correct_64x64x32.npy",
             ImagePath(_dataset="AIBL", _3d=True, _info="orig_patch17"): "AIBL\\orig_getpatch17_correct_64x64x64.npy",
             ImagePath(_dataset="AIBL", _3d=True, _info="orig_patch19"): "AIBL\\orig_getpatch19_correct_64x64x128.npy",
             ImagePath(_dataset="AIBL", _3d=True, _info="orig_patch19_64x64x80"): "AIBL\\orig_getpatch19_correct_64x64x80.npy",
             ImagePath(_dataset="AIBL", _3d=True, _info="orig_patch27"): "AIBL\\orig_getpatch27_correct_64x64x64.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_coronal_64centerslices"): "AIBL\\orig_coronal_64centerslices.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_coronal_32centerslices"): "AIBL\\orig_coronal_32centerslices.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_coronal_4centerslices"): "AIBL\\orig_coronal_4centerslices.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_axial_64centerslices"): "AIBL\\orig_axial_64centerslices.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_axial_32centerslices"): "AIBL\\orig_axial_32centerslices.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_axial_4centerslices"): "AIBL\\orig_axial_4centerslices.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_sagittal_64centerslices"): "AIBL\\orig_sagittal_64centerslices.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_sagittal_32centerslices"): "AIBL\\orig_sagittal_32centerslices.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_sagittal_4centerslices"): "AIBL\\orig_sagittal_4centerslices.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_coronal_64centerslices_flattened"): "AIBL\\orig_coronal_64centerslices_flattened.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_coronal_32centerslices_flattened"): "AIBL\\orig_coronal_32centerslices_flattened.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_coronal_4centerslices_flattened"): "AIBL\\orig_coronal_4centerslices_flattened.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_axial_64centerslices_flattened"): "AIBL\\orig_axial_64centerslices_flattened.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_axial_32centerslices_flattened"): "AIBL\\orig_axial_32centerslices_flattened.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_axial_4centerslices_flattened"): "AIBL\\orig_axial_4centerslices_flattened.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_sagittal_64centerslices_flattened"): "AIBL\\orig_sagittal_64centerslices_flattened.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_sagittal_32centerslices_flattened"): "AIBL\\orig_sagittal_32centerslices_flattened.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="orig_sagittal_4centerslices_flattened"): "AIBL\\orig_sagittal_4centerslices_flattened.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="norm_axial_64centerslices"): "AIBL\\norm_axial_64centerslices.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="norm_axial_32centerslices"): "AIBL\\norm_axial_32centerslices.npy",
             ImagePath(_dataset="AIBL", _3d=False, _info="norm_axial_4centerslices"): "AIBL\\norm_axial_4centerslices.npy",

             ImagePath(_dataset="OASIS",_3d=False, _info="segmented"):"OASIS\\images_hippo_volumes.npy",
             ImagePath(_dataset="OASIS",_3d=True, _info="norm"):"OASIS\\images_3d_norm.npy",
             ImagePath(_dataset="OASIS",_3d=True, _info="orig"):"OASIS\\images_3d_orig.npy",
             ImagePath(_dataset="OASIS", _3d=True, _info="orig_patch26"): "OASIS\\orig_getpatch26_correct_32x64x32.npy",
             ImagePath(_dataset="OASIS", _3d=True, _info="orig_patch26_32x32x64"): "OASIS\\orig_getpatch26_correct_32x32x64.npy",
             ImagePath(_dataset="OASIS", _3d=True, _info="orig_patch6"): "OASIS\\orig_getpatch6_correct_64x64x128.npy",
             ImagePath(_dataset="OASIS", _3d=True, _info="orig_patch16"): "OASIS\\orig_getpatch16_correct_64x64x32.npy",
             ImagePath(_dataset="OASIS", _3d=True, _info="orig_patch17"): "OASIS\\orig_getpatch17_correct_64x64x64.npy",
             ImagePath(_dataset="OASIS", _3d=True, _info="orig_patch19"): "OASIS\\orig_getpatch19_correct_64x64x128.npy",
             ImagePath(_dataset="OASIS", _3d=True, _info="orig_patch19_64x64x80"): "OASIS\\orig_getpatch19_correct_64x64x80.npy",
             ImagePath(_dataset="OASIS", _3d=True, _info="orig_patch27"): "OASIS\\orig_getpatch27_correct_64x64x64.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_coronal_64centerslices"): "OASIS\\orig_coronal_64centerslices.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_coronal_32centerslices"): "OASIS\\orig_coronal_32centerslices.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_coronal_4centerslices"): "OASIS\\orig_coronal_4centerslices.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_axial_64centerslices"): "OASIS\\orig_axial_64centerslices.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_axial_32centerslices"): "OASIS\\orig_axial_32centerslices.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_axial_4centerslices"): "OASIS\\orig_axial_4centerslices.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_sagittal_64centerslices"): "OASIS\\orig_sagittal_64centerslices.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_sagittal_32centerslices"): "OASIS\\orig_sagittal_32centerslices.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_sagittal_4centerslices"): "OASIS\\orig_sagittal_4centerslices.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_coronal_64centerslices_flattened"): "OASIS\\orig_coronal_64centerslices_flattened.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_coronal_32centerslices_flattened"): "OASIS\\orig_coronal_32centerslices_flattened.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_coronal_4centerslices_flattened"): "OASIS\\orig_coronal_4centerslices_flattened.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_axial_64centerslices_flattened"): "OASIS\\orig_axial_64centerslices_flattened.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_axial_32centerslices_flattened"): "OASIS\\orig_axial_32centerslices_flattened.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_axial_4centerslices_flattened"): "OASIS\\orig_axial_4centerslices_flattened.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_sagittal_64centerslices_flattened"): "OASIS\\orig_sagittal_64centerslices_flattened.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_sagittal_32centerslices_flattened"): "OASIS\\orig_sagittal_32centerslices_flattened.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="orig_sagittal_4centerslices_flattened"): "OASIS\\orig_sagittal_4centerslices_flattened.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="norm_axial_64centerslices"): "OASIS\\norm_axial_64centerslices.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="norm_axial_32centerslices"): "OASIS\\norm_axial_32centerslices.npy",
             ImagePath(_dataset="OASIS", _3d=False, _info="norm_axial_4centerslices"): "OASIS\\norm_axial_4centerslices.npy",
             }

labels_dict ={LabelPath(_dataset="HARP", _onehot=True, _all_fields=False): "Harp\\labels.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=False): "Harp\\labels_NOTonehot.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True): "Harp\\labels_all_fields.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True, _info="patch26"):"Harp\\labels_all_fields.npy",  #è giusto così, non servono i labels_patch26: con il nuovo codice ho corretto anche quelli che non venivano prima
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True, _info="patch6"): "Harp\\labels_all_fields.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True, _info="patch16"): "Harp\\labels_all_fields.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True, _info="patch17"): "Harp\\labels_all_fields.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True, _info="patch19"): "Harp\\labels_all_fields.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True, _info="patch27"): "Harp\\labels_all_fields.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True,_info="orig_coronal_64centerslices_flattened"): "Harp\\labels_all_fields_orig_coronal_64centerslices_flattened.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True,_info="orig_coronal_32centerslices_flattened"): "Harp\\labels_all_fields_orig_coronal_32centerslices_flattened.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True,_info="orig_coronal_4centerslices_flattened"): "Harp\\labels_all_fields_orig_coronal_4centerslices_flattened.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True,_info="orig_axial_64centerslices_flattened"): "Harp\\labels_all_fields_orig_axial_64centerslices_flattened.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True,_info="orig_axial_32centerslices_flattened"): "Harp\\labels_all_fields_orig_axial_32centerslices_flattened.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True,_info="orig_axial_4centerslices_flattened"): "Harp\\labels_all_fields_orig_axial_4centerslices_flattened.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True,_info="orig_sagittal_64centerslices_flattened"): "Harp\\labels_all_fields_orig_sagittal_64centerslices_flattened.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True,_info="orig_sagittal_32centerslices_flattened"): "Harp\\labels_all_fields_orig_sagittal_32centerslices_flattened.npy",
              LabelPath(_dataset="HARP", _onehot=False, _all_fields=True,_info="orig_sagittal_4centerslices_flattened"): "Harp\\labels_all_fields_orig_sagittal_4centerslices_flattened.npy",

              LabelPath(_dataset="AIBL", _onehot=True, _all_fields=False): "AIBL\\labels.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=False): "AIBL\\labels_NOTonehot.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True): "AIBL\\labels_all_fields.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True, _info="patch26"): "AIBL\\labels_patch26_all_fields.npy",  #è giusto così, non servono i labels_patch26: con il nuovo codice ho corretto anche quelli che non venivano prima
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True, _info="patch6"): "AIBL\\labels_all_fields.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True, _info="patch16"): "AIBL\\labels_all_fields.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True, _info="patch17"): "AIBL\\labels_all_fields.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True, _info="patch19"): "AIBL\\labels_all_fields.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True, _info="patch27"): "AIBL\\labels_all_fields.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True,_info="orig_coronal_64centerslices_flattened"): "AIBL\\labels_all_fields_orig_coronal_64centerslices_flattened.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True,_info="orig_coronal_32centerslices_flattened"): "AIBL\\labels_all_fields_orig_coronal_32centerslices_flattened.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True,_info="orig_coronal_4centerslices_flattened"): "AIBL\\labels_all_fields_orig_coronal_4centerslices_flattened.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True,_info="orig_axial_64centerslices_flattened"): "AIBL\\labels_all_fields_orig_axial_64centerslices_flattened.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True,_info="orig_axial_32centerslices_flattened"): "AIBL\\labels_all_fields_orig_axial_32centerslices_flattened.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True,_info="orig_axial_4centerslices_flattened"): "AIBL\\labels_all_fields_orig_axial_4centerslices_flattened.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True,_info="orig_sagittal_64centerslices_flattened"): "AIBL\\labels_all_fields_orig_sagittal_64centerslices_flattened.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True,_info="orig_sagittal_32centerslices_flattened"): "AIBL\\labels_all_fields_orig_sagittal_32centerslices_flattened.npy",
              LabelPath(_dataset="AIBL", _onehot=False, _all_fields=True,_info="orig_sagittal_4centerslices_flattened"): "AIBL\\labels_all_fields_orig_sagittal_4centerslices_flattened.npy",

              LabelPath(_dataset="OASIS", _onehot=True, _all_fields=False): "OASIS\\labels.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=False): "OASIS\\labels_NOTonehot.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True): "OASIS\\labels_all_fields.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True, _info="patch26"): "OASIS\\labels_all_fields.npy",  #è giusto così, non servono i labels_patch26: con il nuovo codice ho corretto anche quelli che non venivano prima
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True, _info="patch6"): "OASIS\\labels_all_fields.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True, _info="patch16"): "OASIS\\labels_all_fields.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True, _info="patch17"): "OASIS\\labels_all_fields.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True, _info="patch19"): "OASIS\\labels_all_fields.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True, _info="patch27"): "OASIS\\labels_all_fields.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True, _info="orig_coronal_64centerslices_flattened"): "OASIS\\labels_all_fields_orig_coronal_64centerslices_flattened.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True,_info="orig_coronal_32centerslices_flattened"): "OASIS\\labels_all_fields_orig_coronal_32centerslices_flattened.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True,_info="orig_coronal_4centerslices_flattened"): "OASIS\\labels_all_fields_orig_coronal_4centerslices_flattened.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True,_info="orig_axial_64centerslices_flattened"): "OASIS\\labels_all_fields_orig_axial_64centerslices_flattened.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True,_info="orig_axial_32centerslices_flattened"): "OASIS\\labels_all_fields_orig_axial_32centerslices_flattened.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True,_info="orig_axial_4centerslices_flattened"): "OASIS\\labels_all_fields_orig_axial_4centerslices_flattened.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True,_info="orig_sagittal_64centerslices_flattened"): "OASIS\\labels_all_fields_orig_sagittal_64centerslices_flattened.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True,_info="orig_sagittal_32centerslices_flattened"): "OASIS\\labels_all_fields_orig_sagittal_32centerslices_flattened.npy",
              LabelPath(_dataset="OASIS", _onehot=False, _all_fields=True,_info="orig_sagittal_4centerslices_flattened"): "OASIS\\labels_all_fields_orig_sagittal_4centerslices_flattened.npy",
              }

def load_image_path(_dataset, _3d, _info):
    other=ImagePath(_dataset, _3d, _info)
    for k,v in images_dict.items():
        if k.__eq__(other):
            path = str(home) + str(v)
            if (mac):
                return path.replace("\\", "/")
            return path

def load_labels_path(_dataset,_onehot, _all_fields, _info):
    other = LabelPath(_dataset,_onehot, _all_fields, _info)
    for k,v in labels_dict.items():
        if k.__eq__(other):
            path=str(home)+str(v)
            if (mac):
                return path.replace("\\", "/")
            return path

#print(load_labels_path(_dataset="OASIS", _onehot=False, _all_fields=True))