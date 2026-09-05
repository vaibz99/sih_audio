import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='model_int8.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
scale, zero_point = input_details['quantization']

print("INPUT_SCALE =", scale)
print("INPUT_ZERO_POINT =", zero_point)