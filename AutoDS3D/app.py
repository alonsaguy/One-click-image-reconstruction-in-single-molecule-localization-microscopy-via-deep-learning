

import tkinter as tk
from tkinter import filedialog
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib
matplotlib.use("TkAgg")

import gradio as gr
from func_utils import func1, func2, func3, func4, func5, func6_1, func6_2, func7


output_window = gr.Textbox(label="Output (concise response to button click, see more detailed outputs in the app running terminal)")

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # AutoDS3D

        An automated and enhanced version of DeepSTORM3D (DS3D) for PSF-engineering-based 3D localization microscopy
        
        **Input**: optical parameters, a PSF z-stack, and raw image sequence (from blinking video)
        
        **Output**: a localization list, as well as some intermediate results (figures)
        
        **Operation**: either step by step or one click
        """
    )

    # define inputs
    state = gr.State(value={})  # global container/holder for parameter sharing

    with gr.Row():
        # an accordion input set for optical parameters
        with gr.Accordion("parameter column 1", open=False):
            M = gr.Number(label="★objective magnification M", value=100)
            NA = gr.Number(label="★numeric aperture NA", value=1.45)
            n_immersion = gr.Number(label="★refractive index of immersion medium", value=1.518)
            lamda = gr.Number(label="emission wavelength lamda [um]", value=0.67)
            n_sample = gr.Number(label="★refractive index of sample", value=1.33)
            f_4f = gr.Number(label="focal length of the 4f setup  [um]", value=100e3)
            ps_camera = gr.Number(label="★camera pixel size [um]", value=11)
            ps_BFP = gr.Number(label="■ pixel size of mask plane [um]", value=30)
            device_id = gr.Number(label="GPU id", value=0)

        # calibration z-stack parameters
        with gr.Accordion('parameter column 2', open=False):
            zstack_file = gr.Textbox(label='★z-stack file (within app folder)', value=r'zstack1_-35_20.tif')
            nfp_text = gr.Textbox(label="★z-stack NFPs (start, end, number) [um]", value='-3.5, 2.0, 56')

            # zstack_file = gr.Textbox(label='★z-stack file (within app folder)', value=r'zstack2_-32_18.tif')
            # nfp_text = gr.Textbox(label="★z-stack NFPs (start, end, number) [um]", value='-3.2, 1.8, 51')

            # zstack_file = gr.Textbox(label='★z-stack file (within app folder)', value=r'zstack3_-06_06.tif')
            # nfp_text = gr.Textbox(label="★z-stack NFPs (start, end, number) [um]", value='-0.6, 0.6, 25')

            NFP = gr.Number(label='★NFP [um]', value=1.5)  # 1.5
            # NFP = gr.Number(label='★NFP [um]', value=2.0)
            # NFP = gr.Number(label='★NFP [um]', value=0.7)

            zrange = gr.Textbox(label='★expected z range (z_min, z_max) [um]', value='0, 4')
            # zrange = gr.Textbox(label='★expected z range (z_min, z_max) [um]', value='0, 1.0')

            raw_image_folder = gr.Textbox(label='★image folder (within app folder)', value=r'ims1')
            # raw_image_folder = gr.Textbox(label='★image folder (within app folder)', value=r'ims2')
            # raw_image_folder = gr.Textbox(label='★image folder (within app folder)', value=r'ims3')

            photon_roi = gr.Textbox(label='★SNR detection ROI (r0, c0, r1, c1)', value='0, 0, 40, 40')
            # photon_roi = gr.Textbox(label='★a sparse ROI [r0, c0, r1, c1]', value='0, 0, 80, 80')
            # photon_roi = gr.Textbox(label='★a sparse ROI (r0, c0, r1, c1)', value='80, 80, 100, 100')

            max_pv = gr.Number(label='■ maximum pixel value (MPV)')
            projection_01 = gr.Number(label='0-1 projection')

        # blinking images
        with gr.Accordion('parameter column 3', open=False):
            num_z_voxel = gr.Number(label='★number of voxels in z', value=81)
            training_im_size = gr.Number(label='training image size', value=121)
            us_factor = gr.Number(label='★up-sampling factor (options: 1, 2, 4)', value=1)
            max_num_particles = gr.Number(label='maximum number of particles', value=35)
            num_training_images = gr.Number(label='number of training images', value=10000)
            test_idx = gr.Number(label='■ test image index (inference test)', value=10)
            threshold = gr.Number(label='■ threshold (0-800)', value=40)


    with gr.Row():
        input1 = [M, NA, lamda, n_immersion, n_sample, f_4f, ps_camera, ps_BFP, zstack_file, nfp_text,
                  NFP, zrange, device_id, state]
        button1 = gr.Button("characterize PSF")
        button1.click(func1, inputs=input1, outputs=output_window)

        input2 = [raw_image_folder, state]
        button2 = gr.Button("preprocess blinking images")
        button2.click(func2, inputs=input2, outputs=output_window)

        button3 = gr.Button("characterize SNR")
        button3.click(func3, inputs=[photon_roi, max_pv, state], outputs=output_window)

    with gr.Row():
        button4 = gr.Button("generate training data")
        button4.click(func4, inputs=[num_z_voxel, training_im_size, us_factor, max_num_particles,
                                     num_training_images, projection_01, state], outputs=output_window)

        button5 = gr.Button("train localization net")
        button5.click(func5, inputs=[state], outputs=output_window)

    with gr.Row():
        button6_0 = gr.Button("test inference")
        button6_0.click(func6_1, inputs=[threshold, test_idx, state], outputs=output_window)

        button6_1 = gr.Button("localize")
        button6_1.click(func6_2, inputs=[threshold, state], outputs=output_window)


    input7 = [M, NA, lamda, n_immersion, n_sample, f_4f, ps_camera, ps_BFP, zstack_file, nfp_text, NFP, zrange, device_id,
          raw_image_folder, photon_roi, max_pv, num_z_voxel, training_im_size, us_factor,
          max_num_particles, num_training_images, projection_01, test_idx, threshold, state]
    button7 = gr.Button("ONE CLICK", variant='huggingface')
    button7.click(func7, inputs=input7, outputs=output_window)

    output_window.render()



def choose_file():
    """Open a file dialog and let the user choose a file."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(title="Select a file")

    if file_path:
        print(f"Selected file: {file_path}")
        return file_path
    else:
        print("No file selected.")
        return None

def onselect(eclick, erelease):
    """
    Callback function for RectangleSelector.
    eclick  : MouseEvent at start of selection (mouse press)
    erelease: MouseEvent at end of selection (mouse release)
    """
    global roi_coords
    x1, y1 = int(eclick.xdata), int(eclick.ydata)  # Top-left corner
    x2, y2 = int(erelease.xdata), int(erelease.ydata)  # Bottom-right corner
    roi_coords = [y1, x1, y2, x2]  # r0 c0 r1 c1
    print(f'SNR detection ROI (r0, c0, r1, c1): {roi_coords[0]}, {roi_coords[1]}, {roi_coords[2]}, {roi_coords[3]}')


if __name__ == "__main__":
    # file_path = choose_file()
    # if file_path is not None:
    #     image = io.imread(file_path)
    #     fig, ax = plt.subplots()
    #     ax.imshow(image)  # Display the image
    #     roi_coords = []  # Store ROI coordinates
    #     rect_selector = RectangleSelector(ax, onselect, interactive=True, button=[1])
    #     plt.show()
    demo.launch(share=False)  # local computer

    # demo.launch(share=False, server_name="132.68.109.79")  # remote server




