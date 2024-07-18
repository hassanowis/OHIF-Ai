
# Adding a pretrained Nn-Unet model


### Step 1: Adding new Trainer & Configs Files

1. Create a new file in the `trainers` folder, for example, `segmentation_neuroblastoma.py`, following the structure of `segmentation_spleen.py`.
2. Create a new file in the `configs` folder, for example, `segmentation_neuroblastoma.py`, following the structure of `segmentation_spleen.py`.
### Important: These files will be created, but they won't be used during inference since we will utilize the Nn-Unet inference pipeline.


### Step 2: Adding a New Inferer File

1. Create a new file in the `infers` folder, for example, `segmentation_neuroblastoma.py`, following the structure of `segmentation_spleen.py`.

2. Set up the OS environment variables for nnUNet folders (see [nnUNet Setting up Paths](https://github.com/Gitsamshi/nnUNet-1/blob/master/documentation/setting_up_paths.md)). Update the paths to match your actual folder locations:
 ```python
os.environ['nnUNet_preprocessed']= 'OHIF/monailabel/nnunet/Preprocessed'
os.environ['nnUNet_results']= 'OHIF/monailabel/nnunet/Results'
os.environ['nnUNet_raw']= 'OHIF/monailabel/nnunet/Raw'
```

3. From `batchgenerators` & `nnunetv2` import :
 ```python
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results
```
4. Override the `run_inferer` method from the parent class `monailabel.tasks.infer.basic_infer.BasicInferTask` and add the `nnUNetPredictor` method as shown: 

  ```python
  def run_inferer(self, data, convert_to_batch=True, device="cuda"):
        """
        Run Inferer over pre-processed Data.  Derive this logic to customize the normal behavior.
        In some cases, you want to implement your own for running chained inferers over pre-processed data

        :param data: pre-processed data
        :param convert_to_batch: convert input to batched input
        :param device: device type run load the model and run inferer
        :return: updated data with output_key 
        """
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            device=torch.device('cuda'),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
        )
        
        temp_path= '/temp'
        # Initializes the network architecture, loads the checkpoint
        # Update the paths to your desired model configuration and fold/folds number.
        predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, 'Dataset200_blastoma/nnUNetTrainer__nnUNetPlans__3d_fullres'),
            use_folds=(0,),
            checkpoint_name='checkpoint_best.pth',
        )

        seg=predictor.predict_from_files([[data[self.input_key]]],
                                     temp_path,    
                                     save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=1
                                     num_processes_segmentation_export=1,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
        
        
        if device.startswith("cuda"):
        torch.cuda.empty_cache()

        file_ending='.nii.gz' 
        basename = os.path.basename(data[self.input_key])[:-(len(file_ending) + 5)] + file_ending
        output_path=join(temp_path, basename)
        
        if os.path.exists(output_path):
            outputs = nib.load(output_path).get_fdata()
            outputs = torch.from_numpy(outputs)
            os.remove(output_path)
 
        data[self.output_label_key] = outputs

        return data
  ```

  5. The methods `pre_transforms`, `inferer`, `inverse_transforms`, and `post_transforms` must be overridden. However, they will not be utilized during inference as we will be using the Nn-Unet inference pipeline instead.
e.g: 
```python

    def pre_transforms(self, data=None) -> Sequence[Callable]:
         return []

    
    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(
            roi_size=[128, 128, 32],sw_batch_size = 2, overlap = 0.5
        )

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
            return []
```


