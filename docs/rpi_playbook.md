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

## TF Lite (todo)

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
