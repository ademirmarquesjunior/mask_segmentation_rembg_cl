import os
# from shutil import copyfile
# import sys
from PIL import Image
import numpy as np
import io
from rembg import remove

import argparse
from tqdm import tqdm
from multiprocessing import Process, Pool


# home_folder = os.path.expanduser('~')

# if os.path.exists(os.path.join(str(home_folder) + '\.u2net')) == False:
#     os.makedirs(os.path.join(str(home_folder) + '\.u2net'))
#     copyfile(os.path.normpath(r'u2net.onnx'), os.path.normpath(home_folder + r'\.u2net\u2net.onnx'))

def parallel_core_function(input_path, output_path):
    # Use rembg to remove the background and generate a mask with transparency
    with open(input_path, "rb") as f_in:
        jpg_data = f_in.read()
        mask_data = remove(jpg_data)

    # Convert the mask with transparency to a binary mask
    mask_image = Image.open(io.BytesIO(mask_data))
    binary_mask = create_binary_mask(mask_image)

    # Save the binary mask as a PNG
    binary_mask.save(output_path)
    return


def create_binary_mask(mask_image):
    # Convert the mask to a NumPy array and extract the alpha channel
    mask_np = np.array(mask_image)
    alpha_channel = mask_np[:, :, 3]

    # Threshold the alpha channel to create a binary mask
    threshold = 128  # You can adjust this threshold value if needed
    binary_mask_np = (alpha_channel > threshold).astype(np.uint8) * 255
    return Image.fromarray(binary_mask_np)


class Create_mask():

    def __init__(self, input_dir, output_dir, processes, parent=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.processes = processes

    def process(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        jpg_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith('.jpg')]

        # processes = list()
        list_input = []
        list_output = []
        for jpg_file in jpg_files:
            input_path = os.path.join(self.input_dir, jpg_file)
            mask_filename = os.path.splitext(jpg_file)[0] + "_mask.png"
            output_path = os.path.join(self.output_dir, mask_filename)
            list_input.append(input_path)
            list_output.append(output_path)
            # p = Process(target=parallel_core_function, args=(input_path, output_path))
            # processes.append(p)
            # p.start()

        # for p in processes:
        #     p.join()

        with Pool(self.processes) as p:
            p.starmap(parallel_core_function, zip(list_input, list_output))



if __name__ == '__main__':
    parser = argparse.ArgumentParser( prog='VizLab_rembg', 
                                    description='Program for segmenting background creating mask of jpg images.',
                                    epilog='Developed by Vizlab')



    parser.add_argument('-i', '--inputfolder', required=True)
    parser.add_argument('-o', '--outputfolder', required=True)
    parser.add_argument('-p', '--processes', default=8)

    args = parser.parse_args()

    try:
        input_folder = args.inputfolder
        output_folder = args.outputfolder
        processes = int(args.processes)

        mask_generator = Create_mask(input_folder, output_folder, processes)
        mask_generator.process()
    except Exception as e:
        print('Error: ' + str(e))