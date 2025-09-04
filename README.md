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

This part contains two Jupyter notebooks:
- **Training**: Train your own Deep-STORM model
- **Inference**: Run AutoDS for one-click image reconstruction

> Pre-trained models (4 total) are available in the `models/` directory.

---

## AutoDS3D Instructions

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
 









To run the 3D GUI tool:

1. **Set target machine**  
   In `app.py`, specify whether you're running locally or on a remote server by commenting/uncommenting the appropriate line and setting the IP address.

2. **Set up Python environment**  
   Use a virtual environment with the required packages (see `requirements.txt` if provided).

3. **Run the app**  
   Execute `app.py` (e.g., via terminal or PyCharm) and open the generated URL in your browser.

4. **Fill in parameters**  
   Default values are provided and can be safely used — you can fine-tune them during later steps.

5. **Characterize PSF**  
   After clicking, check:
   - `phase_retrieval_results.jpg` for model accuracy
   - `PSFs.jpg` to verify Z-range and NFP  
   Adjust pixel size and re-run if necessary.

6. **Preprocess blinking images**  
   A confirmation notification will be shown after completion.

7. **Characterize SNR**  
   After clicking:
   - Note the detected MPV (Maximum Pixel Value)
   - For stronger simulated signals, manually set a higher MPV
   - Setting MPV to 0 uses the detected value

8. **Generate training data**  
   Validate the generated training data and adjust MPV as needed.

9. **Train localization network**  
   Launch the training process directly from the GUI.

10. **Run localization test**  
    - Set test image index and threshold (0–800)
    - Check output files:
      - `sim_loc_gt_rec.jpg`
      - `sim_im_gt_rec.jpg`
      - `exp_im_gt_rec.jpg`

11. **Run final localization**  
    Generates a complete localization table of your input data.

---

## Citation

If you use this work, please cite:  
**[One-click image reconstruction in single-molecule localization microscopy via deep learning (bioRxiv link)](LINK_HERE)**
