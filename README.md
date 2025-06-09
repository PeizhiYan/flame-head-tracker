<h1 align="center"><b>FLAME Head Tracker</b></h1>

<div align="center"> 
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <br>
  <span style="font-size:1.0em;">Note: This project may depend on other third-party libraries or code, which may be licensed under different terms. When using this project, you are required to comply with the license terms of any dependencies in addition to the MIT License. Please review the licenses of all dependencies before use or distribution.</span>
</div>

<div align="center"> 
  <b><img src="./assets/demo.gif" alt="drawing" width="600"/></b>
  <br>
</div>

**Current Version**: v3.3 (May 30, 2025)

**Previous Versions**:
- v3.2 Stable (https://github.com/PeizhiYan/flame-head-tracker/tree/v3.2)

## Supported Features:

| Scenario                        | 🙂 Landmarks-based Fitting  | 🔆 Photometric Fitting  |
|---------------------------------|-----------------------------|--------------------------|
| 📷 Single-Image Reconstruction | ✅  | ✅  |
| 🎥 Monocular Video Tracking    | ✅  | Not support yet |



---









## 🦖 Usage


<details>
  <summary><span style="font-size:1.3em">Single-Image-Based Reconstruction 📷</span></summary>

Please follow the example in: ```./Example 1 - single-image reconstruction.ipynb```

```python
from tracker_base import Tracker

tracker_cfg = {
    'mediapipe_face_landmarker_v2_path': './models/face_landmarker.task',
    'flame_model_path': './models/FLAME2020/generic_model.pkl',
    'flame_lmk_embedding_path': './models/landmark_embedding.npy',
    'ear_landmarker_path': './models/ear_landmarker.pth', # this is optional, if you do not want to use ear landmarks during fitting, just remove this line
    'tex_space_path': './models/FLAME_albedo_from_BFM.npz',
    'face_parsing_model_path': './models/79999_iter.pth',
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

- **vertices** `(1, 5023, 3)`  
  The reconstructed FLAME mesh vertices (including expression).  
- **shape** `(1, 300)`  
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
- **cam** `(1, 6)`  
  The estimated 6DoF camera pose (yaw, pitch, roll, x, y, z).  
- **img_rendered** `(1, 256, 256, 3)`  
  Rendered shape on top of the original image (for visualization purposes only).  
- **mesh_rendered** `(1, 256, 256, 3)`  
  Rendered mesh shape with landmarks (for visualization purposes only).  
- **img** `(1, 512, 512, 3)`  
  The image on which the FLAME model was fit.  
- **img_aligned** `(1, 512, 512, 3)`  
  The aligned image.  
- **parsing** `(1, 512, 512)`  
  The face semantic parsing result of `img`.  
- **parsing_aligned** `(1, 512, 512)`  
  The face semantic parsing result of `img_aligned`.  
- **lmks_68** `(1, 68, 2)`  
  The 68 Dlib format face landmarks.  
- **blendshape_scores** `(1, 52)`  
  The facial expression blendshape scores from Mediapipe. 

</details>


---


<details>
  <summary><span style="font-size:1.3em">Monocular Video-Based Tracking 🎥</span></summary>

Please follow the example in: ```./Example 2 - video tracking.ipynb```

```python
from tracker_video import track_video

tracker_cfg = {
    # tracker base settings
    'mediapipe_face_landmarker_v2_path': './models/face_landmarker.task',
    'flame_model_path': './models/FLAME2020/generic_model.pkl',
    'flame_lmk_embedding_path': './models/landmark_embedding.npy',
    'ear_landmarker_path': './models/ear_landmarker.pth', # this is optional, if you do not want to use ear landmarks during fitting, just remove this line
    'tex_space_path': './models/FLAME_albedo_from_BFM.npz',
    'face_parsing_model_path': './models/79999_iter.pth',
    'template_mesh_file_path': './models/head_template.obj',
    'result_img_size': 512,
    'device': device,

    # tracker video settings
    'original_fps': 60,       # input video fps
    'subsample_fps': 30,      # subsample fps
    'video_path': './assets/IMG_2647.MOV',  # example video
    'save_path': './output',  # tracking result save path
}

## Note that, the first frame will take longer time to process
track_video(tracker_cfg)
```

The results will be saved to the ```save_path```. The reconstruction result of each frame will be saved to the corresponding ```[frame_id].npy``` file. 

</details>



<div align="left"> 
  <b>More Examples</b>
  <br>
  <b><img src="./assets/output_MVI_1797.gif" alt="drawing" width="500"/></b>
  <br>
  <b><img src="./assets/output_MVI_1811.gif" alt="drawing" width="500"/></b>
  <br>
  <b><img src="./assets/output_bala.gif" alt="drawing" width="500"/></b>
  <br>
</div>









## 🖥️ Environment Setup


### Prerequisites:

- **GPU**: Nvidia GPU (recommend >= 8GB memory). I tested the code on Nvidia A6000 (48GB) GPU.
- **OS**: Ubuntu Linux (tested on 22.04 LTS and 24.04 LTS), I haven't tested the code on Windows.

### 1️⃣ Step 1: Create a conda environment. 

```
conda create --name tracker -y python=3.10
conda activate tracker
```

### 2️⃣ Step 2: Install necessary libraries.

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




### 3️⃣ Step 3: Download necessary model files.

Because of copyright concerns, we cannot re-share some model files. Please follow the instructions to download the necessary model file.


#### FLAME 

- Download **FLAME 2020 (fixed mouth, improved expressions, more data)** from https://flame.is.tue.mpg.de/ and extract to ```./models/FLAME2020```
    - As an alternative to manually downloading, you can run ```./download_FLAME.sh``` to automatically download and extract the model files.

- Follow https://github.com/TimoBolkart/BFM_to_FLAME to generate the ```FLAME_albedo_from_BFM.npz``` file and place at ```./models/FLAME_albedo_from_BFM.npz```


#### DECA

- Download ```deca_model.tar``` from https://docs.google.com/uc?export=download&id=1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje, and place at ```./models/deca_model.tar```

- Download the files from: https://github.com/yfeng95/DECA/tree/master/data, and place at ```./models/```


#### MICA

- Download ```mica.tar``` from https://drive.google.com/file/d/1bYsI_spptzyuFmfLYqYkcJA6GZWZViNt, and place at ```./models/mica.tar```


#### Mediapipe Face Landmarker

- Download ```face_landmarker.task``` from https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task, rename as ```face_landmarker_v2_with_blendshapes.task```, and save at ```./models/face_landmarker.task```

#### Ear Landmarker (Optional)

If you want to use ear landmarks during the fitting, please download our pre-trained ear landmarker model ```ear_landmarker.pth``` from https://github.com/PeizhiYan/flame-head-tracker/releases/download/resource/ear_landmarker.pth, and save at ```./models/```. But note that, this mode was trained on the i-Bug ear landmarks dataset, which is for RESEARCH purpose ONLY.


The final structure of ```./models/``` is:

```
./models
    ├── 79999_iter.pth                 <----- face parsing model
    ├── deca_model.tar                 <----- deca model
    ├── ear_landmarker.pth             <----- our ear landmarker model
    ├── face_landmarker.task           <----- mediapipe face landmarker model
    ├── fixed_displacement_256.npy
    ├── FLAME2020                      <----- FLAME 2020 model folder
    │   ├── female_model.pkl
    │   ├── generic_model.pkl
    │   ├── male_model.pkl
    │   └── Readme.pdf
    ├── FLAME_albedo_from_BFM.npz      <----- FLAME texture model from BFM_to_FLAME
    ├── head_template.obj              <----- FLAME head template mesh
    ├── landmark_embedding.npy
    ├── mean_texture.jpg
    ├── mica.tar                       <----- mica model
    ├── placeholder.txt
    ├── texture_data_256.npy
    ├── uv_face_eye_mask.png
    └── uv_face_mask.png
```
















## ⚖️ Acknowledgement and Disclaimer

### Acknowledgement

Our code is mainly based on the following repositories:

- FLAME: https://github.com/soubhiksanyal/FLAME_PyTorch
- Pytorch3D: https://github.com/facebookresearch/pytorch3d
- DECA: https://github.com/yfeng95/DECA
- MICA: https://github.com/Zielon/MICA
- FLAME Photometric Fitting: https://github.com/HavenFeng/photometric_optimization
- FaceParsing: https://github.com/zllrunning/face-parsing.PyTorch
- Dlib2Mediapipe: https://github.com/PeizhiYan/Mediapipe_2_Dlib_Landmarks
- Face Alignment: https://github.com/1adrianb/face-alignment
- i-Bug Ears (ear landmarks dataset): https://ibug.doc.ic.ac.uk/resources/ibug-ears/
- Ear Landmark Detection: https://github.com/Dryjelly/Face_Ear_Landmark_Detection
- ArcFace (from InsightFace): https://github.com/deepinsight/insightface

We want to acknowledge the contributions of the authors of these repositories. We do not claim ownership of any code originating from these repositories, and any modifications we have made are solely for our specific use case. All original rights and attributions remain with the respective authors.

### Disclaimer

Our code can be used for research purposes, **provided that the terms of the licenses of any third-party code, models, or dependencies are followed**. For commercial use, the parts of code we wrote are for free, but please be aware to get permissions from any third-party to use their code, models, or dependencies. We do not assume any responsibility for any issues, damages, or liabilities that may arise from the use of this code. Users are responsible for ensuring compliance with any legal requirements, including licensing terms and conditions, and for verifying that the code is suitable for their intended purposes.





## 🧸 Citation

Please consider citing our work if you find this code useful. This code was originally used for "Gaussian Deja-vu" (accepted for WACV 2025 in Round 1). 

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


## Star History

<div align="center">
  <a href="https://www.star-history.com/#PeizhiYan/flame-head-tracker&Date">
    <img src="https://api.star-history.com/svg?repos=PeizhiYan/flame-head-tracker&type=Date" width="500"/>
  </a>
</div>
