# CSC340_AI_Momo_coin_detector_paradise
This project is use to submit in _CSC340 Artificial Intelligence_ class, at _SIT@KMUTT, Thailand_
Provide only trained model and execute file, if you want to use model please install Required Python library and setup tensorflow guide below

## Required Python library
Using ```pip install <name>```
1. hdf5
2. opencv-python
3. tensorflow (or tensorflow-gpu)
4. pandas
5. pillow
6. comtypes
7. numpy
8. matplotlib
9. contextlib2

## Setup tensorflow to use this model
1. Setup [Tensorflow Model](https://github.com/tensorflow/models) to your machine
2. **(If)** you want to use CUDA to help program to precess please install CUDA and cuDNN which is support to your tensorflow-gpu version
   * For tensorflow-gpu 1.14
     * Visual Studio C++ 2015
     * CUDA 10.0
     * cuDNN 10.0
3. Drag and drop these into your tensorflow **\models\research\object_detection\**
<img src="/docs/01.JPG">
4. Execute by running

```python momo_call.py```
    
## More details about PyGrabber
[andreaschiavinato/python_grabber](https://github.com/andreaschiavinato/python_grabber)

## To train your own model
This Git repository provide only trained model and execute file.
If you want to use please following this guide
[EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#2-set-up-tensorflow-directory-and-anaconda-virtual-environment)
