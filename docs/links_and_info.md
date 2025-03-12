# Translations

* Jay — сойка
* Jackdaw — галка
* Magpie - сорока
* Crow - крук
* Tit - синиця
* Tutledove - горлиця

Not seen yet:

* Bullfinch - снігур

# Sounds

## Magpie

* https://www.youtube.com/watch?v=Kz4SvP0c_VE

The best!

* https://www.british-birdsongs.uk/magpie/

# Amazing! - BirdNET

* [Audio data processing](https://github.com/kahst/BirdNET-Analyzer).
* [Live map of bird sound detections](https://app.birdweather.com/) — over 1000 stations world-wide.

# RPi hardware

* SD classes [explained](https://www.pcworld.com/article/2583549/sd-card-terms-specs-explained-sdxc-uhs-v90.html)
* NVMe [characteristics](https://www.enterprisestorageforum.com/hardware/how-fast-are-nvme-speeds/).
* Recording [audio](https://www.circuitbasics.com/how-to-record-audio-with-the-raspberry-pi/)
* [Controlling Arduino](https://www.circuitbasics.com/using-raspberry-pi-to-control-arduino-with-firmata/)

# TFLite

* [Docs](https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter)
* In use — [2.16.1](https://github.com/tensorflow/tensorflow/tree/v2.16.1)
* [Converter](https://android.googlesource.com/platform/external/tensorflow/+/ec63214f098a2bfc87b628219ad0718750d4e930/tensorflow/lite/g3doc/guide/get_started.md)

## Examples

* https://github.com/raspberrypi/picamera2/tree/main/examples/tensorflow

# Models

* [YOLOv5](https://www.kaggle.com/models/kaggle/yolo-v5) for TFLite.
  * Compiled to `_edgetpu.tflite` but it's useless, with zero (0) operations supported by TPU. See https://coral.ai/docs/edgetpu/models-intro/#model-requirements
* TFLite [exporter](https://github.com/zldrobit/yolov5/tree/tf-android)?
