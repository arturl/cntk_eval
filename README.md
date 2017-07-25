# cntk_eval
Image recognition using CNTK

This example shows how to perform image recognition using CNTK

*Note*

Special instructions for building OpenCV-based app:

1. OpenCV must be consumed via vcpkg

2. Download file `https://github.com/martinwicke/tensorflow-tutorial/raw/master/tensorflow_inception_graph.pb` and put in Assets directory of the OpenCV sample app

To build console project with VS 2017:

Remove ` And '$(PlatformToolset.ToLower())' == 'v140'` from the `ImageRecognition\packages\CNTK.CPUOnly.2.0.0\build\native\CNTK.CPUOnly.targets` file
