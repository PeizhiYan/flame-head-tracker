<h1 align="center"><b>FLAME Head Tracker</b></h1>

<div align="center"> 
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
</div>

<div align="center"> 
  <b><img src="./assets/demo-new.gif" alt="drawing" width="600"/></b>
  <br>
  (First two example videos were from IMavatar: <a href="https://github.com/zhengyuf/IMavatar">https://github.com/zhengyuf/IMavatar</a><br><span style='color:lime'>Green Dots</span>: 68 Face Landmarks; <span style='color:blue'>Blue Dots</span>: 10 Eye Landmarks; <span style='color:magenta'>Pink Dots</span>: 40 Ear Landmarks.)
</div>

**Last Major Update**: Dec.-26-2024 🎅
**Version**: 3.2 (March-27-2025)


## Supported Features:

- Mediapipe face landmark detector
- Ear landmarks
- Face parsing masks
- Adjustable camera FOV (currently support image-based reconstruction only)
- Outputs in easy-to-understand format

| Scenario                        | 🙂 Landmarks-based Fitting  | 🔆 Photometric Fitting  |
|---------------------------------|-----------------------------|--------------------------|
| 📷 Single-Image Reconstruction | ✅ Yes | ✅ Yes |
| 📸 Multi-View Reconstruction   | ✅ Yes | ✅ Yes |
| 🎥 Monocular Video Tracking    | ✅ Yes | Not support yet |



---





## 🧸 Citation

This code was originally used for "Gaussian Deja-vu" (accepted for WACV 2025 in Round 1). Please consider citing our work if you find this code useful.

```bibtex
@InProceedings{Yan_2025_WACV,
    author    = {Yan, Peizhi and Ward, Rabab and Tang, Qiang and Du, Shan},
    title     = {Gaussian Deja-vu: Creating Controllable 3D Gaussian Head-Avatars with Enhanced Generalization and Personalization Abilities},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {276-286}
}
```








## 🦖 Usage


<details>
  <summary><b>📷 Single-Image-Based Reconstruction 📷</b></summary>

Please follow the example in: ```./Example_single-image-reconstruction.ipynb```

```python
from tracker_base import Tracker

tracker_cfg = {
    'mediapipe_face_landmarker_v2_path': './models/face_landmarker_v2_with_blendshapes.task',
    'flame_model_path': './models/FLAME2020/generic_model.pkl',
    'flame_lmk_embedding_path': './models/landmark_embedding.npy',
    'ear_landmarker_path': './models/ear_landmarker.pth', # this is optional, if you do not want to use ear landmarks during fitting, just remove this line
    'tex_space_path': './models/FLAME_albedo_from_BFM.npz',
    'face_parsing_model_path': './utils/face_parsing/79999_iter.pth',
    'template_mesh_file_path': './models/head_template.obj',
    'result_img_size': 512,
    'device': device,
}

tracker = Tracker(tracker_cfg)

tracker.update_fov(fov=20)                 # optional setting
#tracker.update_fov(fov=50)                # optional setting (better for selfie images)
#tracker.set_landmark_detector('FAN')      # optional setting
tracker.set_landmark_detector('mediapipe') # optional setting


ret_dict = tracker.load_image_and_run(img_path, realign=True, photometric_fitting=False)
```

The result ```ret_dict``` contains the following data:

- **vertices** `(5023, 3)`  
  The reconstructed FLAME mesh vertices (including expression).  
- **shape** `(1, 100)`  
  The FLAME shape code.  
- **exp** `(1, 100)`  
  The FLAME expression code.  
- **pose** `(1, 6)`  
  The FLAME head (first 3 values) and jaw (last 3 values) poses.  
- **eye_pose** `(1, 6)`  
  The FLAME eyeball poses.  
- **tex** `(1, 50)`  
  The FLAME parametric texture code.  
- **light** `(1, 9, 3)`  
  The estimated SH lighting coefficients.  
- **cam** `(6,)`  
  The estimated 6DoF camera pose (yaw, pitch, roll, x, y, z).  
- **img_rendered** `(256, 256, 3)`  
  Rendered shape on top of the original image (for visualization purposes only).  
- **mesh_rendered** `(256, 256, 3)`  
  Rendered mesh shape with landmarks (for visualization purposes only).  
- **img** `(512, 512, 3)`  
  The image on which the FLAME model was fit.  
- **img_aligned** `(512, 512, 3)`  
  The aligned image.  
- **parsing** `(512, 512, 3)`  
  The face semantic parsing result of `img`.  
- **parsing_aligned** `(512, 512, 3)`  
  The face semantic parsing result of `img_aligned`.  
- **lmks_dense** `(478, 2)`  
  The 478 dense face landmarks from Mediapipe.  
- **lmks_68** `(68, 2)`  
  The 68 Dlib format face landmarks.  
- **blendshape_scores** `(52,)`  
  The facial expression blendshape scores from Mediapipe. 


### Example Reconstruction Result (realign=True):

![](./assets/single_image_fitting_1.png)

### Example Reconstruction Result (realign=False):

![](./assets/single_image_fitting_2.png)

</details>


---


<details>
  <summary><b>📸 Multi-View Reconstruction 📸</b></summary>

Please follow the example in: ```./Example_multi-view-reconstruction.ipynb```


The result ```ret_dict``` contains the following data:

- **vertices** `(5023, 3)`  
  The reconstructed canonical FLAME mesh vertices (including expression).  
- **shape** `(1, 100)`  
  The FLAME canonical shape code.  
- **exp** `(1, 100)`  
  The FLAME canonical expression code.  
- **pose** `(1, 6)`  
  The FLAME canonical head (first 3 values) and jaw (last 3 values) poses.  
- **eye_pose** `(1, 6)`  
  The FLAME canonical eyeball poses.  
- **tex** `(1, 50)`  
  The FLAME canonical parametric texture code.  
- **light** `(1, 9, 3)`  
  The estimated canonical SH lighting coefficients.  
- **cam** `N*(6,)`  
  The estimated 6DoF camera pose (yaw, pitch, roll, x, y, z) for each view.  
- **img_rendered** `N*(256, 256, 3)`  
  Rendered shapes on top of the original images (for visualization purposes only).  
- **mesh_rendered** `N*(256, 256, 3)`  
  Rendered mesh shapes with landmarks (for visualization purposes only).  
- **img** `N*(512, 512, 3)`  
  The images (views) on which the FLAME model was fit.  
- **img_aligned** `N*(512, 512, 3)`  
  The aligned images.  
- **parsing** `N*(512, 512, 3)`  
  The face semantic parsing results of `img`.  
- **parsing_aligned** `N*(512, 512, 3)`  
  The face semantic parsing results of `img_aligned`.

</details>

---

<details>
  <summary><b>🎥 Monocular Video-Based Tracking 🎥</b></summary>

Please follow the example in: ```./Example_video-reconstruction.ipynb```

```python
from tracker_video import track_video

tracker_cfg = {
    'mediapipe_face_landmarker_v2_path': './models/face_landmarker_v2_with_blendshapes.task',
    'flame_model_path': './models/FLAME2020/generic_model.pkl',
    'flame_lmk_embedding_path': './models/landmark_embedding.npy',
    'ear_landmarker_path': './models/ear_landmarker.pth', # this is optional, if you do not want to use ear landmarks during fitting, just remove this line
    'tex_space_path': './models/FLAME_albedo_from_BFM.npz',
    'face_parsing_model_path': './utils/face_parsing/79999_iter.pth',
    'template_mesh_file_path': './models/head_template.obj',
    'result_img_size': 512,
    'device': device,
    ## following are used for video tracking
    'original_fps': 60,       # input video fps
    'subsample_fps': 30,      # subsample fps
    'video_path': './assets/IMG_2647.MOV',  # example video
    'save_path': './output',  # tracking result save path
    'use_kalman_filter': False, # whether to use Kalman filter
    'kalman_filter_measurement_noise_factor': 1e-5, # measurement noise level in Kalman filter 
    'kalman_filter_process_noise_factor': 1e-5,     # process noise level in Kalman filter 
}

## Note that, the first frame will take longer time to process
track_video(tracker_cfg)
```

The results will be saved to the ```save_path```. The reconstruction result of each frame will be saved to the corresponding ```[frame_id].npy``` file. 

</details>












## 🖥️ Environment Setup


<details>
  <summary><b>Details</b></summary>

### Prerequisites:

- **GPU**: Nvidia GPU with >= 6GB memory (recommend > 8GB). I tested the code on Nvidia A6000 (48GB) GPU.
- **OS**: Ubuntu Linux (tested on 22.04 LTS and 24.04 LTS), I haven't tested the code on Windows.

### Step 1: Create a conda environment. 

```
conda create --name tracker -y python=3.10
conda activate tracker
```

### Step 2: Install necessary libraries.

#### Nvidia CUDA compiler (11.7)

```
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit ninja

# (Linux only) ----------
ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"  # to avoid error "/usr/bin/ld: cannot find -lcudart"

# Install NVCC (optional, if the NVCC is not installed successfully try this)
conda install -c conda-forge cudatoolkit=11.7 cudatoolkit-dev=11.7
```

After install, check NVCC version (should be 11.7):

```
nvcc --version
```

#### PyTorch (2.0 with CUDA)

```
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu117
```

Now let's test if PyTorch is able to access CUDA device, the result should be ```True```:

```
python -c "import torch; print(torch.cuda.is_available())"
```

#### Some Python packages

```
pip install -r requirements.txt
```

#### Nvidia Differentiable Rasterization: nvdiffrast

**Note that**, we use nvdiffrast version **0.3.1**, other versions may also work but not promised.

```
# Download the nvdiffrast from their official Github repo
git clone https://github.com/NVlabs/nvdiffrast

# Go to the downloaded directory
cd nvdiffrast

# Install the package
pip install .

# Change the directory back
cd ..
```

#### Pytorch3D

**Note that**, we use pytorch3d version **0.7.8**, other versions may also work but not promised.

Installing pytorch3d may take a bit of time.

```
# Download Pytorch3D from their official Github repo
git clone https://github.com/facebookresearch/pytorch3d

# Go to the downloaded directory
cd pytorch3d

# Install the package
pip install .

# Change the directory back
cd ..
```

#### Troubleshoot

Note that the NVCC needs g++ < 12:
```
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 50
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 50
sudo update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-11 50
```

If there is problem with **nvdiffrast**, check whether it is related to the EGL header file in the error message. If it is, install the EGL Development Libraries (for Ubuntu/Debian-based systems):
```
sudo apt-get update
sudo apt-get install libegl1-mesa-dev
```
Then, uninstall nvdiffrast and reinstall it.


### Step 3: Download some necessary model files.

Because of copyright concerns, we cannot re-share any of the following model files. Please follow the instructions to download the necessary model file.

#### FLAME and DECA

- Download ```FLAME 2020 (fixed mouth, improved expressions, more data)``` from https://flame.is.tue.mpg.de/ and extract to ```./models/FLAME2020```
- Download the files from: https://github.com/yfeng95/DECA/tree/master/data, and place at ```./models/```
- Follow https://github.com/TimoBolkart/BFM_to_FLAME to generate the ```FLAME_albedo_from_BFM.npz``` file and place at ```./models/```
- Download ```deca_model.tar``` from https://docs.google.com/uc?export=download&id=1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje, and place at ```./models/```

#### Mediapipe

- Download ```face_landmarker.task``` from https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task, rename as ```face_landmarker_v2_with_blendshapes.task```, and save at ```./models/```

#### Ear Landmarker

If you want to use ear landmarks during the fitting, please download our pre-trained ear landmarker model ```ear_landmarker.pth``` from https://github.com/PeizhiYan/flame-head-tracker/releases/download/resource/ear_landmarker.pth, and save at ```./models/```. But note that, this mode was trained on the i-Bug ear landmarks dataset, which is for RESEARCH purpose ONLY.


The final structure of ```./models/``` is:

```
./models
    ├── deca_model.tar
    ├── face_landmarker_v2_with_blendshapes.task
    ├── fixed_displacement_256.npy
    ├── FLAME2020
    │   ├── female_model.pkl
    │   ├── generic_model.pkl
    │   ├── male_model.pkl
    │   └── Readme.pdf
    ├── FLAME_albedo_from_BFM.npz
    ├── head_template.obj
    ├── landmark_embedding.npy
    ├── mean_texture.jpg
    ├── placeholder.txt
    ├── texture_data_256.npy
    ├── uv_face_eye_mask.png
    └── uv_face_mask.png
    └── ear_landmarker.pth
```

</details>






## Troubleshoot

<details>
<summary><b>Cuda error (with Nvdiffrast)</b></summary>
  <b>If you observe error message like the following:</b>
  <p>
    File "/home/peizhi/miniconda3/envs/tracker/lib/python3.10/site-packages/nvdiffrast/torch/ops.py", line 246, in forward
      out, out_db = _get_plugin(gl=True).rasterize_fwd_gl(raster_ctx.cpp_wrapper, pos, tri, resolution, ranges, peeling_idx)
    RuntimeError: Cuda error: 219[cudaGraphicsGLRegisterBuffer(&s.cudaPosBuffer, s.glPosBuffer, cudaGraphicsRegisterFlagsWriteDiscard);]
  </p>
  <p>
    <b>Potential Solution:</b>
    Your system might have multiple CUDA GPUs. In that case, try changing the GPU device specified in your code to see if it resolves the problem. For example, if you originally set <i>device='cuda:0'</i> and encountered the error, try setting it to <i>device='cuda:1'</i> instead.
  </p>
</details>








## ⚖️ Acknowledgement and Disclaimer

### Acknowledgement

Our code is mainly based on the following repositories:

- FLAME: https://github.com/soubhiksanyal/FLAME_PyTorch
- Nvdiffrast: https://github.com/NVlabs/nvdiffrast
- Pytorch3D: https://github.com/facebookresearch/pytorch3d
- DECA: https://github.com/yfeng95/DECA
- FLAME Photometric Fitting: https://github.com/HavenFeng/photometric_optimization
- 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
- GaussianAvatars: https://shenhanqian.github.io/gaussian-avatars
- FaceParsing: https://github.com/zllrunning/face-parsing.PyTorch
- Dlib2Mediapipe: https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks
- Face Alignment: https://github.com/1adrianb/face-alignment
- i-Bug Ears (ear landmarks dataset): https://ibug.doc.ic.ac.uk/resources/ibug-ears/
- Ear Landmark Detection: https://github.com/Dryjelly/Face_Ear_Landmark_Detection

We want to acknowledge the contributions of the authors of these repositories. We do not claim ownership of any code originating from these repositories, and any modifications we have made are solely for our specific use case. All original rights and attributions remain with the respective authors.

### Disclaimer

Our code can be used for research purposes, **provided that the terms of the licenses of any third-party code, models, or dependencies are followed**. For commercial use, the parts of code we wrote are for free, but please be aware to get permissions from any third-party to use their code, models, or dependencies. We do not assume any responsibility for any issues, damages, or liabilities that may arise from the use of this code. Users are responsible for ensuring compliance with any legal requirements, including licensing terms and conditions, and for verifying that the code is suitable for their intended purposes.





## 📃 Todos
<details>
  <summary><b>Todo List</b></summary>

- [x] Improve video tracking speed. (addressed in v1.01)  
- [x] Add Kalman filter for temporal camera pose smoothing. (addressed in v1.1)  
- [x] Add support for photometric fitting. (addressed in v2.0)  
- [x] Add support for multi-view fitting. (addressed in v2.1)  
- [x] Add ear landmarks detection module, and include ear landmarks during the fitting process. (addressed in v3.0)
- [ ] Add symmetric constraints to ear landmarks-guided fitting (when the other ear is not visible).
- [ ] Add dynamic ear landmark loss weights based on the head pose.
- [ ] Temporal smooth in the face alignment and cropping.
- [ ] Improve efficiency and reconstruction quality.

</details>


## Star History

<div align="center">
  <a href="https://www.star-history.com/#PeizhiYan/flame-head-tracker&Date">
    <img src="https://api.star-history.com/svg?repos=PeizhiYan/flame-head-tracker&type=Date" width="500"/>
  </a>
</div>
