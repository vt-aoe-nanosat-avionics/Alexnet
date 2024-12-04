import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

def get_flops(model):
  forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
  graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
  flops = graph_info.total_float_ops
  return flops


model = tf.keras.models.load_model("models/model.keras")


# model.compile(optimizer='adam', loss='bce', metrics=['accuracy'])

flops = get_flops(model)
macs = flops / 2
print(f"MACs: {macs:,}")
print(f"FLOPs: {flops:,}")