# Intro

## Goal

To run model inference on the M.2 Dual Edge TPU [Accelerator](https://coral.ai/docs/m2/get-started/) with Python.

Coral [tech specs](https://coral.ai/products/m2-accelerator-ae/#tech-specs).

## Given

- `.tflite` models are the models that run on the Edge devices (smartphones, Raspberry PIs, etc.). They are smaller and more energy-efficient.
- For Edge **TPUs**, we need another type of a `.tflite` model, [compiled](https://coral.ai/docs/edgetpu/compiler/#system-requirements) specifically for it. They typically have a `_edgetpu.tflite` suffix.
- Coral M.2 Edge TPU was made compayible with TFlite (Tensorflow Lite) API ([link](https://coral.ai/docs/edgetpu/tflite-python/)) so, in an ideal world, I should be able to run a `.tflite` model on Edge TPU with [minimal modifications](https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu).


## Need:

[This](https://coral.ai/docs/notes/build-coral/#required-components) is the best description of all required components I found so far.
And [this](https://coral.ai/docs/m2/get-started/#2a-on-linux) is a vanilla setup & check guide (only partially helpful).

1. M.2 [setup](https://coral.ai/docs/m2/get-started/)

    Checks:

    ```bash
    # Verify that the accelerator module is detected:

    lspci -nn | grep 089a

    # This Should print 2 devices. E.g.:
    0000:03:00.0 System peripheral [0880]: Global Unichip Corp. Coral Edge TPU [1ac1:089a]
    0000:04:00.0 System peripheral [0880]: Global Unichip Corp. Coral Edge TPU [1ac1:089a]

    # Verify that the PCIe driver (gasket-dkms) is loaded:

    ls /dev | grep apex

    # Should echo back "/dev/apex_0"

    # Verify the Edge TPU Runtime (libedgetpu) is installed:

    python3 -c "from ctypes.util import find_library; print(find_library(\"edgetpu\"))"

    # should echo back "libedgetpu.so.1"

    ```

2. Edge TPU [compiler](https://coral.ai/docs/edgetpu/compiler/#download) to compile a regular `.tflite` model to `_edgetpu.tflite` format.

    Checks:

    ```bash
    edgetpu_compiler [options] model...
    ```

3. TFLite runtime

    a. Again, not compatible with Python > 3.9
    b. Can try to cross-compile from x86_64 to ARM64 (there is no supported toolchain for compiling ARM-to-ARM)
    c. Should be compiled from the same `TENSORFLOW_COMMIT` as the rest of the libraries.
    d. Test:

        ```python
        from tflite_runtime.interpreter import Interpreter, load_delegate

        interpreter = Interpreter(model_path='efficientdet_lite0_edgetpu.tflite',experimental_delegates=[load_delegate('libedgetpu.so.1')])
        ```

4. (Optionally) Pycoral library.

    a. [deb definition](https://github.com/google-coral/pycoral/blob/master/debian/control)
    b. Compatible only with `Python <=3.9` (`<<3.10`):

        ```bash
        % apt-cache show python3-pycoral
            Package: python3-pycoral
            Source: coral
            Version: 2.0.0
            Installed-Size: 2911
            Maintainer: Coral <coral-support@google.com>
            Architecture: arm64
            Depends: libc6, libedgetpu1-std (= 16.0) | libedgetpu1 (= 16.0), libgcc1, libstdc++6, python3-numpy, python3-pil, python3-tflite-runtime (= 2.5.0.post1), python3 (<< 3.10), python3:any (>= 3.3.2-2~)
            Description: Coral Python API
            Python library for interfacing with the Edge TPU on various Coral products
        ```
    c. Even if I try to build it from sources, it depends on `python3-tflite-runtime (= 2.5.0.post1)`, which also requires Python `<3.9`:

        ```bash
        % apt-cache show python3-tflite-runtime
            Package: python3-tflite-runtime
            Source: tflite-runtime
            Version: 2.5.0.post1
            Installed-Size: 13173
            Maintainer: TensorFlow team <packages@tensorflow.org>
            Architecture: arm64
            Depends: python3 (<< 3.10), python3-numpy (>= 1:1.12.1), python3:any (>= 3.3.2-2~), libc6 (>= 2.14), libgcc1 (>= 1:4.7), libstdc++6 (>= 6)
            Description: TensorFlow Lite is for mobile and embedded devices.
            TensorFlow Lite is the official solution for running machine learning models on
            mobile and embedded devices. It enables on-device machine learning inference
            with low latency and a small binary size on Android, iOS, and other operating
            systems.
        ```

    d. Test:

        ```bash
        # https://coral.ai/docs/notes/build-coral/#test-everything

        mkdir coral && cd coral

        git clone https://github.com/google-coral/tflite.git

        cd tflite/python/examples/classification

        bash install_requirements.sh

        python3 classify_image.py \
        --model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
        --labels models/inat_bird_labels.txt \
        --input images/parrot.jpg
        ```

## Status

1. Both checks pass
2. Cannot install:

    ```bash
    E: Unable to locate package edgetpu-compiler
    ```

    This note explains it?

    > The Edge TPU Compiler is no longer available for ARM64 systems (such as the Coral Dev Board), beginning with version 2.1. We recommend compiling your model on a more powerful desktop.

    **Solution**: Try [web-based compiler](https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb).

3. Cannot install (Py version)
4. Cannot install (Py version)
5. Cannot install - unavailable for `aarch64` architecture ([link](https://pypi.org/project/ai-edge-litert/#files))

## Possible solutions

1. Use older Rasp OS installation that uses `python-3.9` as the default.
2. Install `python-3.9` as system default (Install `3.9` & use `update-alternatives --config python`)
   * Pros: Should be easy?
   * Cons: Potential issues with system libs relying on `3.11` features; Going backward - losing access to new features, performance and security updates.
     (e.g., Python 3.11 is said to be between 10-60% [faster than 3.10](https://docs.python.org/3/whatsnew/3.11.html#summary-release-highlights))
   * Links:
     * https://thelinuxcode.com/change-from-default-to-alternative-python-version-debian/
     * https://pythonspeed.com/articles/base-image-python-docker-images/
3. Use Docker image inside the RPi.
   * Pros: Should be easy?
     * Not sure I completely understand how will it work with the device drivers. Privileged access to host? If host provides drivers, then it makes no sense as I can't install drivers on host correctly due to versions mismatch. If container controls drivers then it needs to control host's _kernel?_ Brrr!
   * Cons: Complicates development; Potential issues with making drivers work correctly; Adding a layer of virtualization will affect performance.
4. Try to [build](https://coral.ai/docs/notes/build-coral/#build-the-libcoral-api) `pycoral` and other packages from from sources.
   * Pros: Full control in case of success; Keeping same dev process; Staying on host avoids another layer of virtualization.
   * Cons: Potential rabbit hole of unimaginable depth.
   * Links:
     * `tflite_runtime` [build script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh)
       (see [also](./compile_tflite_runtime.sh))
     * `gasket_driver` [build instructions](https://github.com/google/gasket-driver/blob/main/README.md)
     * `libedgetpu` [build instructions](https://github.com/google-coral/libedgetpu/blob/master/makefile_build/README.md)
5. Use `pyenv` with `3.9` and use only PyPi libs, not .deb. Which means no `pycoral` but `tflite_runtime` will work.
   * Pros: easy
   * Cons: Tried it, it didn't work with `edgetpu` ("no delegates found error"). Potentially because the `gasket-dkms` package is also not compatible with `3.11` and I had to build it from the sources. Maybe it's functional or maybe it's not (all surface driver checks pass but how to verify it's really working?)
6. Migrate to [Mendel Linux](https://coral.ai/software/#mendel-linux)?
   * Unfortunately, only [suppors](https://coral.googlesource.com/docs/+/refs/heads/master/ReadMe.md#what-do-we-support) Coral Dev Board.
7. Migrate off Coral to Hailo NPU (AI Accelerator)?
   * Pros:
      * May entirely solve all stupid issues I have. The single biggest root of all issues I have with loading the TPU delegate is that Google abandoned its product (again. surprise-surprise!), - there is almost no up-to-date documentation, the drivers are 5+ years old, `tflite_runtime` is deprecated in favor of new shiny `litert` lib (which doesn't support ARM), and so on (the rant list is big here).
      * More power and better efficiency. 13 TOPS vs. 8 (2x4), 4 TOPS/W vs. 2 TOPS/W. Yikes!
      * Promises to work with any framework: TF, TFLite, PyTorch, Keras, etc.:
       _Comprehensive dataflow compiler enables customer to port their neural network models easily & quickly_ ([link](https://hailo.ai/products/ai-accelerators/hailo-8l-ai-accelerator-for-ai-light-applications/#hailo8l-features))
      * Comprehensive [model garden](https://hailo.ai/products/hailo-software/model-explorer/)?
   * Cons:
      * A lot of unknowns. It's a new product with an unknown quality of documentation, their SDK, etc. Some real chance to get stuck when converting some experimental model, etc.
      * I will have to assemble another Pi (~250 EUR)
   * Links:
      * [Product page](https://hailo.ai/products/ai-accelerators/)
      * [RPi](https://www.raspberrypi.com/products/ai-kit/)
      * [M2 form factors](https://www.dell.com/support/kbdoc/en-us/000144170/how-to-distinguish-the-differences-between-m-2-cards), https://www.easeus.com/computer-instruction/m2-2230-vs-2280.html
      * [Software support](https://hailo.ai/products/hailo-software/hailo-ai-software-suite/#sw-modelzoo).
      * Buy:
        * RPi AI Hat: https://buyzero.de/en/products/raspberry-pi-ai-hat-13-26-tops
        * Combo Ai Hat + NVM: https://buyzero.de/en/products/pineboards-ai-bundle-hailo-8l-nvme-2280-fur-raspberry-pi-5


## Other

* Coral software - https://coral.ai/software/#edgetpu-runtime
* Related issue — https://github.com/google-coral/edgetpu/issues/777



## Installing the driver for TPU (still ongoing)

```sh
#!/usr/bin/env sh

# The official [installation steps](https://coral.ai/docs/m2/get-started#4-run-a-model-on-the-edge-tpu) didn't
# work for me - I was having postinstall errors from `dkms`:
# dkms: running auto installation service for kernel 6.1.0-26-arm64.
# dkms: autoinstall for kernel: 6.1.0-26-arm64.
# Setting up linux-headers-arm64 (6.1.112-1) ...
# Setting up gasket-dkms (1.0-18) ...
# locale: Cannot set LC_CTYPE to default locale: No such file or directory
# locale: Cannot set LC_ALL to default locale: No such file or directory
# Loading new gasket-1.0 DKMS files...
# Deprecated feature: REMAKE_INITRD (/usr/src/gasket-1.0/dkms.conf)
# Building for 6.6.51+rpt-rpi-2712 6.6.51+rpt-rpi-v8
# Building initial module for 6.6.51+rpt-rpi-2712
# Deprecated feature: REMAKE_INITRD (/var/lib/dkms/gasket/1.0/source/dkms.conf)
# Error! Bad return status for module build on kernel: 6.6.51+rpt-rpi-2712 (aarch64)
# Consult /var/lib/dkms/gasket/1.0/build/make.log for more information.
# dpkg: error processing package gasket-dkms (--configure):
#  installed gasket-dkms package post-installation script subprocess returned error exit status 10
# Processing triggers for man-db (2.11.2-2) ...
# Processing triggers for libc-bin (2.36-9+rpt2+deb12u8) ...
# Errors were encountered while processing:
#  gasket-dkms
# E: Sub-process /usr/bin/dpkg returned an error code (1)
#
# So I adapted the script kindly provided by @dataslayermedia's [gist](https://gist.github.com/dataslayermedia/714ec5a9601249d9ee754919dea49c7e),
# and it worked.
# The main difference with the official installation guide I can see, is that the `dkms` dependencies are installed
# in "layers", and the dkms driver package is built from the GitHub repo, instead of trying to install all at once
# as the `gasket-dkms` package.
# I also skipped all steps that were doing manupulations with the Device Tree Blob — the setup seems to be working
# fine without these.

# Clean up previous installation
sudo apt-get remove -y gasket-dkms
sudo apt-get remove -y libedgetpu1-std

sudo apt update
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update

sudo apt install -y devscripts debhelper

sudo apt install -y dkms

sudo apt-get install -y dh-dkms

sudo apt-get install -y libedgetpu1-std

sudo git clone https://github.com/google/gasket-driver.git
cd gasket-driver/
sudo debuild -us -uc -tc -b

cd ..
sudo dpkg -i gasket-dkms_1.0-18_all.deb

sudo sh -c "echo 'SUBSYSTEM==\"apex\", MODE=\"0660\", GROUP=\"apex\"' >> /etc/udev/rules.d/65-apex.rules"
echo "kernel=kernel8.img" | sudo tee -a /boot/firmware/config.txt

# Verify the installation (you may need to run `sudo reboot` first)
# This should show 2 apex devices
ls /dev/apex*
# Double-check that the accelerator module is enabled (You should see something like "03:00.0 System peripheral: Device 1ac1:089a")
lspci -nn | grep 089a
```

## TF Lite

### Edge TPU Compiler

`tflite` != `edgetpu`. A `tflite` model need to be converted using the Edge TPU compiler into a file that's compatible with the Edge TPU.
See the Coral [page](https://coral.ai/docs/edgetpu/compiler/).

* Only available on Debian and x86_64 architecture.
* Use [web-based](https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb) compiler

### TensorFlow Lite Runtime: Libs

Tried to `pip install tflite-support` [library](https://pypi.org/project/tflite-support/) used in the official Tensorflow [tutorial](https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/requirements.txt) — failed, as the only available version was some 0.1.1 alpha:

```sh
% pip install tflite-support
Looking in indexes: https://pypi.org/simple, https://www.piwheels.org/simple
Collecting tflite-support
  Downloading tflite-support-0.1.0a1.tar.gz (390 kB)
  Preparing metadata (setup.py) ... done
```

(nothing worked there)

A forced installation with specific version didn't work too:

```sh
% pip install "tflite-support>=0.4.2"
Looking in indexes: https://pypi.org/simple, https://www.piwheels.org/simple
ERROR: Could not find a version that satisfies the requirement tflite-support>=0.4.2 (from versions: 0.1.0a0.dev3, 0.1.0a0.dev4, 0.1.0a0.dev5, 0.1.0a0, 0.1.0a1)
ERROR: No matching distribution found for tflite-support>=0.4.2
```

### tflite-runtime

Claude said "a-ha! that's because ..." and pointed to another tflite [library](https://pypi.org/project/tflite-runtime/).
I don't understand the difference between them - the description is almost identical to `tflite-support` and names itself `TensorFlow Lite`:

```text
TensorFlow Lite is the official solution for running machine learning models on mobile and embedded devices. It enables on-device machine learning inference with low latency and a small binary size on Android, iOS, and other operating systems.
```

Even worse, `tflite-support` and `tflite-runtime` had their last release 1+ year ago, and approximately at the same time (Jul'23 and Oct'23)

### ai-edge-litert

This is what [is recommended](https://ai.google.dev/edge/litert/inference#run-python) by Google as a replacement for `tflite`. Didn't install for me either:

```sh
% pip install ai-edge-litert
Looking in indexes: https://pypi.org/simple, https://www.piwheels.org/simple
ERROR: Could not find a version that satisfies the requirement ai-edge-litert (from versions: none)
ERROR: No matching distribution found for ai-edge-litert
```

As with previous libs, this one is also perfectly [available](https://pypi.org/project/ai-edge-litert/) on PyPi so, at this point, I started to suspect that maybe they are not being built for RPi/ARM, despite being visually available at the package index.

### Distributions

Indeed, it looks that for Python 3.11/ARM combination, the `ai-edge-litert` lib is available only for Mac OS, plus in 2x `x86_64` distributions for Linux/Mac.
The `tflite-support` lib is not available for Py 3.11/ARM at all, only for `x86_64`:

[link](https://pypi.org/project/ai-edge-litert/#files):

```text
Built Distributions
ai_edge_litert-1.0.1-cp311-cp311-manylinux_2_17_x86_64.whl (2.2 MB view details)
Uploaded Aug 30, 2024 CPython 3.11 manylinux: glibc 2.17+ x86-64

ai_edge_litert-1.0.1-cp311-cp311-macosx_12_0_arm64.whl (2.3 MB view details)
Uploaded Aug 30, 2024 CPython 3.11 macOS 12.0+ ARM64

ai_edge_litert-1.0.1-cp311-cp311-macosx_10_15_x86_64.whl (2.8 MB view details)
Uploaded Aug 30, 2024 CPython 3.11 macOS 10.15+ x86-64

ai_edge_litert-1.0.1-cp310-cp310-manylinux_2_17_x86_64.whl (2.2 MB view details)
Uploaded Aug 30, 2024 CPython 3.10 manylinux: glibc 2.17+ x86-64

ai_edge_litert-1.0.1-cp310-cp310-macosx_12_0_arm64.whl (2.3 MB view details)
Uploaded Aug 30, 2024 CPython 3.10 macOS 12.0+ ARM64

ai_edge_litert-1.0.1-cp310-cp310-macosx_10_15_x86_64.whl (2.8 MB view details)
Uploaded Aug 30, 2024 CPython 3.10 macOS 10.15+ x86-64

ai_edge_litert-1.0.1-cp39-cp39-manylinux_2_17_x86_64.whl (2.2 MB view details)
Uploaded Aug 30, 2024 CPython 3.9 manylinux: glibc 2.17+ x86-64

ai_edge_litert-1.0.1-cp39-cp39-macosx_12_0_arm64.whl (2.3 MB view details)
Uploaded Aug 30, 2024 CPython 3.9 macOS 12.0+ ARM64

ai_edge_litert-1.0.1-cp39-cp39-macosx_10_15_x86_64.whl (2.8 MB view details)
Uploaded Aug 30, 2024 CPython 3.9 macOS 10.15+ x86-64
```

```text
Built Distributions
tflite_support-0.4.4-cp311-cp311-manylinux2014_x86_64.whl (60.8 MB view details)
Uploaded Jul 12, 2023 CPython 3.11

tflite_support-0.4.4-cp311-cp311-macosx_10_11_x86_64.whl (58.3 MB view details)
Uploaded Jul 12, 2023 CPython 3.11 macOS 10.11+ x86-64

tflite_support-0.4.4-cp310-cp310-manylinux2014_x86_64.whl (60.8 MB view details)
Uploaded Jul 12, 2023 CPython 3.10

tflite_support-0.4.4-cp310-cp310-macosx_10_11_x86_64.whl (58.3 MB view details)
Uploaded Jul 12, 2023 CPython 3.10 macOS 10.11+ x86-64

tflite_support-0.4.4-cp39-cp39-manylinux2014_x86_64.whl (60.8 MB view details)
Uploaded Jul 12, 2023 CPython 3.9

tflite_support-0.4.4-cp39-cp39-manylinux2014_armv7l.whl (33.2 MB view details)
Uploaded Jul 12, 2023 CPython 3.9

tflite_support-0.4.4-cp39-cp39-manylinux2014_aarch64.whl (43.1 MB view details)
Uploaded Jul 12, 2023 CPython 3.9

tflite_support-0.4.4-cp39-cp39-macosx_10_11_x86_64.whl (58.3 MB view details)
Uploaded Jul 12, 2023 CPython 3.9 macOS 10.11+ x86-64

tflite_support-0.4.4-cp38-cp38-manylinux2014_armv7l.whl (33.2 MB view details)
Uploaded Jul 12, 2023 CPython 3.8

tflite_support-0.4.4-cp38-cp38-manylinux2014_aarch64.whl (43.1 MB view details)
Uploaded Jul 12, 2023 CPython 3.8

tflite_support-0.4.4-cp37-cp37m-manylinux2014_armv7l.whl (33.2 MB view details)
Uploaded Jul 12, 2023 CPython 3.7m

tflite_support-0.4.4-cp37-cp37m-manylinux2014_aarch64.whl (43.1 MB view details)
Uploaded Jul 12, 2023 CPython 3.7m
```

Bottom line:

* `ai-edge-litert` — 100% ❌ (no ARM distributions for RPi)
* `tflite-litert` - Try with Python 3.9 (`tflite_support-0.4.4-cp39-cp39-manylinux2014_armv7l.whl`)


## PyEnv

[Automatic installation](https://github.com/pyenv/pyenv?tab=readme-ov-file#1-automatic-installer-recommended):

```sh
curl https://pyenv.run | bash
```

## Compiling Python 3.9

```sh
# Solves build issue: "ERROR: The Python ssl extension was not compiled". Installs 'openssl' and few other libs as its dependancies.
sudo apt-get install -y libssl-dev libffi-dev
```
```sh
# Solves build warnings:
#  * No module named '_bz2'
#  * No module named '_curses'
#  * No module named '_sqlite3'
#  * No module named 'readline'
#  * No module named '_lzma'
sudo apt-get install -y \
    libbz2-dev \
    libncurses5-dev libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev
```
```sh
pyenv install 3.9
```

A clean installation this time! ✨

## Back to TFLite

Refresh shell -> `pyenv virtualenv tflite` -> `pyenv activate tflite`, then:

```sh
pip install "tflite-support>=0.4.2"
# Installing collected packages: flatbuffers, pycparser, pybind11, protobuf, numpy, absl-py, CFFI, sounddevice, tflite-support
# Successfully installed CFFI-1.17.1 absl-py-2.1.0 flatbuffers-20181003210633 numpy-2.0.2 protobuf-3.20.3 pybind11-2.13.6 pycparser-2.22 sounddevice-0.5.1 tflite-support-0.4.4
pip install "opencv-python-headless==4.10.0.84"
# Using cached opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl (29.3 MB)
# Installing collected packages: opencv-python-headless
pip install "numpy==1.26.4"
#     Uninstalling numpy-2.0.2:
#       Successfully uninstalled numpy-2.0.2
# Successfully installed numpy-1.26.4
```

Another clean installation! ✨

#### Test

```python
import cv2

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
```

#### Test next

```python
# ... imports

# Initialize the object detection model
model = "efficientdet_lite0.tflite"
enable_edgetpu = True
num_threads = 4

base_options = core.BaseOptions(
    file_name=model,
    use_coral=enable_edgetpu,
    num_threads=num_threads,
)
detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    detection_options=detection_options,
)

detector = vision.ObjectDetector.create_from_options(options)
print(detector)
```

error:

```text
File "~/home/val~/.pyenv/versions/3.9.20/envs/tflite/lib/python3.9/site-packages/tensorflow_lite_support/python/task/vision/object_detector.py", line 90, in create_from_options
    detector = _CppObjectDetector.create_from_options(
RuntimeError: Plugin did not create EdgeTpuCoral delegate.
```

### Fixing "Plugin did not create EdgeTpuCoral delegate."

**Checklist:**

- [x] The [Edge TPU Runtime](https://coral.ai/software/#edgetpu-runtime) installed (`libedgetpu1-std`)
- [x] The Coral driver (`gasket-dkms`) — installed, `1.0-18`


## Pipx / Hatch (skipped)

- [pipx](https://github.com/pypa/pipx)

```sh
sudo apt-get update && \
    sudo apt-get install -y pipx && \
    pipx ensurepath && \
    sudo pipx ensurepath --global
```

- [hatch](https://hatch.pypa.io/1.12/install/#pipx)

```sh
pipx install hatch
```
