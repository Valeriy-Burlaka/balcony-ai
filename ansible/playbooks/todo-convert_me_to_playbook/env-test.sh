
verify_tensorfow_installation() {
    # TF
    python3 -c "import tensorflow as tf; print('Random tensor test result: ', tf.reduce_sum(tf.random.normal([1000, 1000])))"

    python3 -c "import tensorflow as tf; print('GPU: ', tf.config.list_physical_devices('GPU'))"

    python3 -c "import tensorflow as tf; print('CPU: ', tf.config.list_physical_devices('CPU'))"

    # TF-Lite runtime
    python3 -c "from tflite_runtime.interpreter import Interpreter, load_delegate"
}


# import tensorflow
#from tflite_runtime.interpreter import Interpreter, load_delegate


# interpreter = Interpreter(model_path='efficientdet_lite0_edgetpu.tflite',experimental_delegates=[load_delegate('libedgetpu.so.1')])
# interpreter = Interpreter(model_path='efficientdet_lite0_edgetpu.tflite',experimental_delegates=[load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')])
# interpreter.allocate_tensors()