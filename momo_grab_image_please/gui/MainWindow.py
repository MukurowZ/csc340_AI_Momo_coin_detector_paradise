#
# python_grabber
#
# Authors:
#  Andrea Schiavinato <andrea.schiavinato84@gmail.com>
#
# Copyright (C) 2019 Andrea Schiavinato
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

import queue
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import csv
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog

from gui.SelectDevice import *
from gui.ConfigureRecording import *
from pygrabber.PyGrabber import *
from pygrabber.image_process import *

sys.path.append("..")
sys.path.append("../..")

from utils import label_map_util
from utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()

class MainWindow:
    def __init__(self, master):
        self.create_gui(master)
        self.grabber = PyGrabber(self.on_image_received)
        self.queue = queue.Queue()
        self.image = None
        self.original_image = None
        self.select_device()

    def create_gui(self, master):
        self.master = master
        master.title("Momo coin detection - Prototype")
        self.create_menu(master)

        master.columnconfigure(0, weight=1, uniform="group1")
        master.columnconfigure(1, weight=1, uniform="group1")
        master.rowconfigure(0, weight=1)

        self.video_area = Frame(master, bg='black')
        self.video_area.grid(row=0, column=0, sticky=W+E+N+S, padx=5, pady=5)

        self.status_area = Frame(master)
        self.status_area.grid(row=1, column=0, sticky=W+E+N+S, padx=5, pady=5)

        self.image_area = Frame(master)
        self.image_area.grid(row=0, column=1, sticky=W+E+N+S, padx=5, pady=5)

        self.image_controls_area = Frame(master)
        self.image_controls_area.grid(row=1, column=1, padx=5, pady=0)

        self.image_controls_area2 = Frame(master)
        self.image_controls_area2.grid(row=2, column=1, padx=5, pady=0)

        # Grabbed image
        fig = Figure(figsize=(5, 4), dpi=100)
        self.plot = fig.add_subplot(111)
        self.plot.axis('off')

        self.canvas = FigureCanvasTkAgg(fig, master=self.image_area)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=1)

        # Status
        self.lbl_status1 = Label(self.status_area, text="No device selected")
        self.lbl_status1.grid(row=0, column=0, padx=5, pady=5, sticky=W)

        # Image controls
        self.grab_btn = Button(self.image_controls_area, text="Grab", command=self.grab_frame)
        self.grab_btn.pack(padx=5, pady=20, side=LEFT)

        self.save_btn = Button(self.image_controls_area2, text="Process", command=self.save_image)
        self.save_btn.pack(padx=5, pady=2, side=LEFT)

        self.video_area.bind("<Configure>", self.on_resize)

    def create_menu(self, master):
        menubar = Menu(master)
        self.master.config(menu=menubar)

        camera_menu = Menu(menubar)
        camera_menu.add_command(label="Open...", command=self.change_camera)
        camera_menu.add_command(label="Set properties...", command=self.camera_properties)
        camera_menu.add_command(label="Start preview", command=self.start_preview)
        menubar.add_cascade(label="Camera", menu=camera_menu)

        image_menu = Menu(menubar)
        image_menu.add_command(label="Grab image", command=self.grab_frame)
        menubar.add_cascade(label="Image", menu=image_menu)

    def display_image(self):
        while self.queue.qsize():
            try:
                self.image = self.queue.get(0)
                self.original_image = self.image
                self.plot.imshow(np.flip(self.image, axis=2))
                self.canvas.draw()
            except queue.Empty:
                pass
        self.master.after(100, self.display_image)

    def select_device(self):
        input_dialog = SelectDevice(self.master, self.grabber.get_video_devices())
        self.master.wait_window(input_dialog.top)
        # no device selected
        if input_dialog.device_id is None:
            exit()

        self.grabber.set_device(input_dialog.device_id)
        self.grabber.start_preview(self.video_area.winfo_id())
        self.display_status(self.grabber.get_status())
        self.on_resize(None)
        self.display_image()

    def display_status(self, status):
        self.lbl_status1.config(text=status)

    def change_camera(self):
        self.grabber.stop()
        del self.grabber
        self.grabber = PyGrabber(self.on_image_received)
        self.select_device()

    def camera_properties(self):
        self.grabber.set_device_properties()

    def set_format(self):
        self.grabber.display_format_dialog()

    def on_resize(self, event):
        self.grabber.update_window(self.video_area.winfo_width(), self.video_area.winfo_height())

    def init_device(self):
        self.grabber.start()

    def grab_frame(self):
        self.grabber.grab_frame()

    def on_image_received(self, image):
        self.queue.put(image)

    def start_preview(self):
        self.grabber.start_preview(self.video_area.winfo_id())
        self.display_status(self.grabber.get_status())
        self.on_resize(None)

    def stop(self):
        self.grabber.stop()
        self.display_status(self.grabber.get_status())

    def save_image(self):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y")
        time_string = now.strftime("%H_%M_%S")
        filename = CWD_PATH + '\momo_grab_image_please\process_data\\' + str(dt_string) + str(time_string) + ".jpg"
        if filename is not None:
            self.process_coin(filename)

    def process_coin(self, filename):
        MODEL_NAME = 'inference_graph'

        coin1 = 0
        coin2 = 0
        coin5 = 0
        coin10 = 0
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y")
        time_string = now.strftime("%H:%M:%S")

        # os.system('python ' + CWD_PATH + '\Object_detection_image.py ' + filename)
        PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

        # Number of classes the object detector can identify
        NUM_CLASSES = 4

        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image_expanded = np.expand_dims(self.image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visulaize the results')

        total_coin = []
        for index, value in enumerate(classes[0]):
            object_dict = {}
            if scores[0, index] > 0.7:
                object_dict[(category_index.get(value)).get('name')] = \
                    scores[0, index]
                print(object_dict)
                total_coin.append([*object_dict])

        print(total_coin)

        for a in total_coin:
            print(a[0])
            if a[0] == '1baht':
                coin1 = coin1 + 1
            if a[0] == '2baht':
                coin2 = coin2 + 1
            if a[0] == '5baht':
                coin5 = coin5 + 1
            if a[0] == '10baht':
                coin10 = coin10 + 1

        coin_counter = open(CWD_PATH+'/coin_counter_momo.csv', "a+")
        coin_counter.write("\n"+ dt_string + ',' + time_string + ',' + str(coin1) + ',' +str(coin2*2) + ',' + str(coin5*5)+ ',' + str(coin10*10))
        coin_counter.close()
        # coin_counter = coin_counter + ( dt_string + ',' + time_string + ',' + str(coin1) + ',' +str(coin2*2) + ',' + str(coin5*5)+ ',' + str(coin10*10) )
        # print(coin_counter)
        # coin_counter.to_csv(CWD_PATH+'/coin_counter_momo.csv')

        vis_util.visualize_boxes_and_labels_on_image_array(
            self.image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

        cv2.imshow('Momo paradise',self.image)

        # Press any key to close the image
        cv2.waitKey(0)

        # Clean up
        cv2.destroyAllWindows()

