# \[EECS 442 Project\] SIM-Sync-Mono (Part 2: SIM-Sync-Mono)

This repository contains a Python implementation of Joint Depth Estimation and Certifiably Optimal Synchronization Using Learned Module, as described in the paper:

`Xihang Yu, Yuchen Zhou, Guoyuan Li. SIM-Sync-Mono: Joint Depth Estimation and Certifiably Optimal Synchronization Using Learned Module. EECS 442 Fall 2023`

###  [Paper](https://xihangyu630.github.io/assets/pdf/SIM_Sync_Mono_final_project.pdf) | [Colab](https://colab.research.google.com/drive/1sE0VmWCuL6HUad3yXHEZoCvJKzAvINpz?authuser=1#scrollTo=_p7km19VLzx3)

We introduce an innovative approach for estimating camera trajectories and 3D scene structures from multiview image keypoints by utilizing a pretrained depth prediction network that efficiently separates camera pose estimation and depth fine-tuning and a SIM-Sync solver to optimally solve the structured problem of camera trajectory estimation. The approach is validated through experiments on the TUM dataset.

![bundle_adjustment](https://drive.google.com/uc?export=view&id=10VHFFIzCttl6t5LkRXRZZ0uXv3dI9G1D)


## Installation

Dependencies: [MiDaS](https://github.com/isl-org/MiDaS), PyTorch, Open3D, TEASER, MOSEK.


## Citation
If you find our work useful in your research, please consider citing:
```BibTeX
@inproceedings{SIM-Sync-Mono,
 title={SIM-Sync-Mono: Joint Depth Estimation and Certifiably Optimal Synchronization Using Learned Module},
 author={Xihang Yu, Yuchen Zhou, Guoyuan Li},
 year={2023},
}
```

## Acknowledgements
We also thank the authors for releasing [robust-cvd](https://robust-cvd.github.io/), [PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch), [MOSEK](https://www.mosek.com/), [open3d](http://www.open3d.org/), [TEASER++](https://github.com/MIT-SPARK/TEASER-plusplus).
