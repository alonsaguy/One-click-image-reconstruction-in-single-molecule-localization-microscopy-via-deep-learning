# One-Click Image Reconstruction in Single-Molecule Localization Microscopy via Deep Learning

![Alt text](intro_image.png)

This repository accompanies the paper:  
**_"One-click image reconstruction in single-molecule localization microscopy via deep learning"_**

> 📄 Please make sure to [cite the original paper](#citation) when using or extending this work.

---

## Repository Structure

This project is divided into two main components:

- **AutoDS** — AutoDS inference and 2D Deep-STORM model training
- **AutoDS3D** — GUI-based AutoDS3D localization pipeline

---

## AutoDS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alonsaguy/One-click-image-reconstruction-in-single-molecule-localization-microscopy-via-deep-learning/blob/main/AutoDS/AutoDS_inference_v1_2.ipynb)

This part contains two Jupyter notebooks:
- **Training**: Train your own Deep-STORM model
- **Inference**: Run AutoDS for one-click image reconstruction

> Pre-trained models (4 total) are available in the `models/` directory.

---

## AutoDS3D Instructions

The main one-click GUI provides easy access for most users. Open-source code is also available for users who want to explore technical details and do customization, along with in-situ PSF modeling.

1, one click GUI: please see **AutoDS3D/guideline.docx** for implementation instructions.

directly linked code and data: app.py, app_utils.py, func_utils.py, dataset1

2, open-source code

directly linked code and data: training.ipynb, inference.ipynb, dataset2

3, in_situ PSF modeling

directly linked code and data: insitu_sim.ipynb, insitu_exp.ipynb, in_situ_data

reference: Liu, Sheng, et al. "Universal inverse modeling of point spread functions for SMLM localization and microscope characterization." Nature Methods 21.6 (2024): 1082-1093.

<br>updates

2025/10/24: open-source code, adjusted the data structure and loss function selection

2025/11/05: from now one, see now updates in: https://github.com/dafeixiao/AutoDS3D



## Citation

If you use this work, please cite:  
**[One-click image reconstruction in single-molecule localization microscopy via deep learning (bioRxiv link)](LINK_HERE)**
