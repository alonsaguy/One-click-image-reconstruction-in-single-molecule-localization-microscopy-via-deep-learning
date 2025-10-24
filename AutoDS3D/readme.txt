
autods3d GUI: app.py, app_utils.py, func_utils.py

open-code of autods3d: train_DS3D.ipynb, infer_DS3D.ipynb, dataset2 [you may need to adjust the path format according to your operating system]

in_situ PSF characterization: insitu_sim.ipynb, insitu_exp.ipynb, in_situ_data




GUI Guideline
0, use anaconda and environment file (environment.yml) to configure an environment for running autods3d. 
1, run app.py to get a local URL and click the URL to launch a web GUI. 
 
Note if you are running this on a server, get into the app.py code and configure the server IP address as follows
 
The web GUI explanation:
 

2, fill parameters in the GUI and process your images with either “ONE CLICK” or step-by-step operation. If all parameters are proper, “ONE CLICK” is legitimate. You can see notifications in the Output window at the bottom of the GUI and some technical notifications appears in the terminal where you run the GUI. 
Step-by-step operation can help tune the parameters.
Characterize PSF: take the bead z-stack file and calibrate the practical PSF model. The PSFs.jpg shows the practical PSF according to NFP and z range and those two parameters should be adjusted such that reasonable PSF shapes are shown in PSFs.jpg.
 
Preprocess images: remove background of images in the image folder and save the preprocessed images in a new folder. All the folder path should be relative to and within the GUI folder.
 
Characterize SNR: take the characterized PSF model and preprocessed images to detect SNR parameters. For an isolated PSF (molecule image) regardless of its shape, the maximum pixel value is defined as MPV which tells, to some extent, the general photon count information. MPV can be tuned according to the feedback coming after clicking the next step--simulate training data. Note that 0 MPV in the GUI means the detected value is used.
  
Simulate training data: the data size is decided by the GUI parameter, number of training images. Although the default value is 10000, it is recommended to start with a smaller number, like 100, for sanity check. The output of this step includes sim-exp.tif which shows one of the simulated images (left) together with one of the experimental images (right), and it’s good for MPV tunning.
 
Train a localization neural net: In addition to typical training results, a package of all the necessary parameters is saved as a pickle file and this pickle file can be filled in the GUI “external training file” to directly apply the trained model in future.
 
Test the net: with both simulated image and experimental image (appointed by test image index). At this step, threshold, which is linked to localization confidence, can be tuned. 
 
Localize all: process all the images and generate a localization list, a csv file.
 











