# TotalSegmentator Integration with MONAILabel

Below are the instructions to set up and use the model.

## Installation

1. Install the required packages:

    ```bash
    pip install totalsegmentator
    ```

## Configuration
### Step 1: Create a New File

Create a new file in the `configs` folder, for example, `total_seg.py`.
The configuration file (`total_seg.py`) includes settings for the TotalSegmentator model. 

### Step 2: Modify the  `init` Method

The `TotalSegmentator` class inherits from `TaskConfig`. You will primarily modify the `init` method:

Set up Labels:
  ```python
  self.labels = {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "pancreas",
        8: "adrenal_gland_right",
        9: "adrenal_gland_left",
        10: "lung_upper_lobe_left",
        11: "lung_lower_lobe_left",
        12: "lung_upper_lobe_right",
        13: "lung_middle_lobe_right",
        14: "lung_lower_lobe_right",
        15: "esophagus",
        16: "trachea",
        17: "thyroid_gland",
        18: "small_bowel",
        19: "duodenum",
        20: "colon",
        21: "urinary_bladder",
        22: "prostate",
        23: "kidney_cyst_left",
        24: "kidney_cyst_right",
        25: "sacrum",
        26: "vertebrae_S1",
        27: "vertebrae_L5",
        28: "vertebrae_L4",
        29: "vertebrae_L3",
        30: "vertebrae_L2",
        31: "vertebrae_L1",
        32: "vertebrae_T12",
        33: "vertebrae_T11",
        34: "vertebrae_T10",
        35: "vertebrae_T9",
        36: "vertebrae_T8",
        37: "vertebrae_T7",
        38: "vertebrae_T6",
        39: "vertebrae_T5",
        40: "vertebrae_T4",
        41: "vertebrae_T3",
        42: "vertebrae_T2",
        43: "vertebrae_T1",
        44: "vertebrae_C7",
        45: "vertebrae_C6",
        46: "vertebrae_C5",
        47: "vertebrae_C4",
        48: "vertebrae_C3",
        49: "vertebrae_C2",
        50: "vertebrae_C1",
        51: "heart",
        52: "aorta",
        53: "pulmonary_vein",
        54: "brachiocephalic_trunk",
        55: "subclavian_artery_right",
        56: "subclavian_artery_left",
        57: "common_carotid_artery_right",
        58: "common_carotid_artery_left",
        59: "brachiocephalic_vein_left",
        60: "brachiocephalic_vein_right",
        61: "atrial_appendage_left",
        62: "superior_vena_cava",
        63: "inferior_vena_cava",
        64: "portal_vein_and_splenic_vein",
        65: "iliac_artery_left",
        66: "iliac_artery_right",
        67: "iliac_vena_left",
        68: "iliac_vena_right",
        69: "humerus_left",
        70: "humerus_right",
        71: "scapula_left",
        72: "scapula_right",
        73: "clavicula_left",
        74: "clavicula_right",
        75: "femur_left",
        76: "femur_right",
        77: "hip_left",
        78: "hip_right",
        79: "spinal_cord",
        80: "gluteus_maximus_left",
        81: "gluteus_maximus_right",
        82: "gluteus_medius_left",
        83: "gluteus_medius_right",
        84: "gluteus_minimus_left",
        85: "gluteus_minimus_right",
        86: "autochthon_left",
        87: "autochthon_right",
        88: "iliopsoas_left",
        89: "iliopsoas_right",
        90: "brain",
        91: "skull",
        92: "rib_left_1",
        93: "rib_left_2",
        94: "rib_left_3",
        95: "rib_left_4",
        96: "rib_left_5",
        97: "rib_left_6",
        98: "rib_left_7",
        99: "rib_left_8",
        100: "rib_left_9",
        101: "rib_left_10",
        102: "rib_left_11",
        103: "rib_left_12",
        104: "rib_right_1",
        105: "rib_right_2",
        106: "rib_right_3",
        107: "rib_right_4",
        108: "rib_right_5",
        109: "rib_right_6",
        110: "rib_right_7",
        111: "rib_right_8",
        112: "rib_right_9",
        113: "rib_right_10",
        114: "rib_right_11",
        115: "rib_right_12",
        116: "sternum",
        117: "costal_cartilages"
  }
```

## Inference

### Step 1: Create a New File

1. Create a new file in the `infers` folder, for example, `total_seg.py`. 
   For inference, use the `TotalSegmentator` defined in `total_seg.py`. This script loads the trained model and applies the necessary transformations to the input data for segmentation.

2. Add the import line in `__init__.py` to import the class that will be built in step 2:
   ```python
   from .total_seg import TotalSegmentator
   ```
### Step 2: Modify the  `init` Method

Create `TotalSegmentator` class that inherits from `BasicInferTask`. This class will define the inference task.

### Step 3: Modify the `run_inferer` Method

```python
def run_inferer(self, data, device="cuda"):

  # Run TotalSegmentator process
  subprocess.run(["TotalSegmentator", "-i", data[self.input_key], "-o", self.temp_path])

  if device.startswith("cuda"):
      torch.cuda.empty_cache()

  # Initialize an empty numpy array for the final label map
  final_label_data = None

  # Loop through the files in the temporary path
  for filename in os.listdir(self.temp_path):
      if filename.endswith(".nii") or filename.endswith(".nii.gz"):
          filepath = os.path.join(self.temp_path, filename)
          label_img = nib.load(filepath)
          label_data = label_img.get_fdata()
          # Determine the label from the filename (assuming the filename matches the labels dictionary values)
          label_name = filename.replace(".nii", "").replace(".gz", "")
          label_value = None
          for key, value in self.labels.items():
              if value == label_name:
                  label_value = key
                  break
          if label_value is not None:
              # Initialize final_label_data if it's the first file
              if final_label_data is None:
                  final_label_data = np.zeros_like(label_data, dtype=int)
              # Assign the label_value to the corresponding regions in the final_label_data
              final_label_data[label_data != 0] = label_value

          # remove the file after processing
          os.remove(filepath)

  # Convert to PyTorch tensor
  final_label_data = torch.from_numpy(final_label_data)

  # Assign the processed label data back to the output label key
  data[self.output_label_key] = final_label_data

  return data
```
