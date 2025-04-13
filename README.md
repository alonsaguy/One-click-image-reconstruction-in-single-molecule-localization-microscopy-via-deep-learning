# One-click-image-reconstruction-in-single-molecule-localization-microscopy-via-deep-learning

This repository is based on the following paper: "One-click image reconstruction in single-molecule localization microscopy via deep learning".

Please cite the original [paper](link with biorXiv) when using or developing this notebook.

The repository is divided to two parts:
- AutoDS
- AutoDS3D

**AutoDS**
There are two jupyter notebooks, one for training our new Deep-STORM models and one for inference using AutoDS pipeline. See run instruction inside the notebooks.

Additionally, we provide a set of four pre-trained models inside the 'models' directory.

 **AutoDS3D**
Instructions for running AutoDS3D via the provided GUI.
1, in app.py file, choose the machine where you want to run this app. If it’s a remote server, comment the line for local computer and give the server IP address, as shown below. 

2, set a python environment

3, run app.py in either a terminal or programing software, e.g. PyCharm to obtain URL. Use the URL in a web browser to enter the web application, as shown below.
 
4, fill in the parameters and feel safe to use default values. Those parameters will be tuned in the following step-by-step operation. 

5, click characterize PSF. You will see the notification below. If the simulation is too small, consider decrease the pixel size of mask plane (with square mark in the web app). In the app folder, check phase_retrieval_results.jpg to verify the model accuracy. Check PSFs.jpg to ensure proper z range and NFP. You can tune these two parameters and re-click characterize PSF to update the imaging model.
 
6, click preprocess blinking images. The notification of this step is 
 
7, click characterize SNR. The notification is shown below. The detected MPV is related the photon count of the simulated training data in the next step. If you want to have stronger signal in training data, set a value bigger than this detected MPV value for maximum pixel value (MPV) in the web app. Note that 0 MPV in the web app means the detected value will be used. 
 
8, click generate training data. check the training data and tune MPV if necessary.
 
9, click training localization net

10, verify test image index (inference test) and threshold (0-800) and then click localization test. Check sim_loc_gt_rec.jpg, sim_im_gt_rec.jpg, and exp_im_gt_rec.jpg for feedback.
Figures

11. click localize. This generates a complete localization table.
