import numpy as np

# Valid ONNX operators supported by both ONNX and Avrex
avrex_data_ops = [ "Identity", "Gemm", "MatMul", "Add" ] # These require data reads
avrex_activ_ops = [ "Relu", "Sigmoid", "Tanh", "Softmax" ] # These are just functions with no data read
avrex_skip_list = [ "Identity", "Flatten", "Reshape" ] # KEEP AWARE THESE WILL CHANGE FOR CNNS
avrex_lut_nodes = ["Sigmoid", "Tanh", "Softmax"]

# Stores onnx name equivalents in avrex
avrex_activ_ops_map = {
    "Relu" : "relu",
    "Sigmoid" : "sigmoid",
    "Tanh" : "tanh",
    "Softmax" : "softmax"
}

# MAKE SURE THESE ARE ACCURATE
avrex_lut_dp_map = {
    "Sigmoid" : 12,
    "Tanh" : 13,
    "Softmax" : 8
}

class AvrexNode:
    def __init__(self, op_type, static_tensors, output_name):
        self.op_type = op_type
        self.static_tensors = static_tensors
        self.output_name = output_name

# Avrex function definitions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def float_to_fixedp(x, dp):
    return int(round(x * (2 ** dp)))

def fixedp_to_float(x, dp):
    return (float(x)) / (2 ** dp)
    
def quantize(x, dp):
    return fixedp_to_float(float_to_fixedp(x, dp), dp)