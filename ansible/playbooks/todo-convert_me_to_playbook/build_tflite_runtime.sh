export CROSSTOOL_PYTHON_INCLUDE_PATH=/usr/include/python3.11
rm -rf tensorflow/lite/tools/pip_package/gen/ > /dev/null && \
tensorflow/lite/tools/pip_package/build_pip_package_with_cmake.sh aarch64 > ../build.log 2>&1 & \
tail -f ../build.log

# Monitor
# watch -n 1 'iostat -x'
# htop