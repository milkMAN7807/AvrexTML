import onnx
from onnx import numpy_helper as nphelper
import numpy as np
import argparse
from cpp_gen import HeaderFile
import avrex_functions as avrfunc
from avrex_functions import avrex_data_ops, avrex_activ_ops, avrex_activ_ops_map
import weight_numerics as quantization
import calibration
import os

parser = argparse.ArgumentParser(description = "Reads a valid ONNX file")
parser.add_argument("--model_path", type = str, help = "Location of ONNX file in directory", required = True)
parser.add_argument("--header_name", type = str, help = "Title of header file (adds .hpp by default)", required = True)
parser.add_argument("--header_output_path", type = str, help = "Path to output headerfile", required = True)
parser.add_argument("--quantize", action = "store_true", help = "Quantizes the model to a ternary format (required if not already done)")
parser.add_argument("--quantize_biases", action = "store_true", help = "Quantizes the model biases to int16 from fp")
parser.add_argument("--input_decimal_point", type = int, default = 0, help = "The fixed decimal point (in binary) of the incoming network data (0 means no decimal)", required = True)
parser.add_argument("--calibration_data_path", type = str, help = "Location of the calibration dataset")
parser.add_argument("--calibrate_decimal_point", action = "store_true", help = "Run calibration data to assign optimal decimal points to each layer, if not just assign equal ranges (8 bits) to each fractional and whole parts")
args = parser.parse_args()

model = onnx.load(args.model_path)
graph = model.graph

# Graph vars
detected_network_inputs = 0
last_node = None # used to check for Matmul/Add pairs to combine into Gemm operations
output_occurence_map = {}
static_tensor_list = []
to_create_list = []

weight_transpose_flag = False # Because models cant make up their mind when deciding weight matrix dimension order

# Grab static data reads
for tensor in graph.initializer:
    static_tensor_list.append(tensor.name)

# Validate graph
for i, node in enumerate(graph.node):
    # Check for constant value node: technically violates topology but in reality... no
    if node.op_type == "Constant":
        static_tensor_list.append(node.output[0])
        continue
    elif node.op_type == "Identity":
        node.output[0] = node.input[0] # Just keep the same i/o (alias it basically so it doesnt create a new one)
        continue

    # Check for invalid Matmul/Add placements
    if (last_node is not None) and (node.op_type == "Add") and (last_node.op_type != "MatMul"):
        raise ValueError(f"Invalid node \"{ node.op_type }\" at index { i }, expected preceding \"MatMul\" node to create an Avrex DenseLayer.")
    elif (node.op_type not in avrex_data_ops) and (node.op_type not in avrex_activ_ops):
        raise ValueError(f"Invalid node \"{ node.op_type }\" at index { i }, this node is not supported by Avrex. Please see readme.")
    
    if len(node.output) > 1:
        raise ValueError(f"Invalid node at index { i }, multiple outputs detected. This is not supported by Avrex. Please see readme.")

    # Update occurences for specific i/o patterns to catch multirouting an output
    for input in node.input:
        if output_occurence_map.get(input) is not None:
            output_occurence_map[input] += 1
            if output_occurence_map[input] > 1:
                raise ValueError("Invalid network, one node is routing to multiple other nodes. It should be routing to only one other. Please see Avrex readme.")
        elif (output_occurence_map.get(input) is None) and (input not in static_tensor_list):
            detected_network_inputs += 1
            output_occurence_map[input] = 1
            
    if detected_network_inputs > 1:
        raise ValueError("Invalid network, multiple entry points detected when there should only be one. Please see Avrex readme.")

    # Validate that last nodes output is being fed into this node
    if (last_node is not None) and set(last_node.output).isdisjoint(node.input):
        raise ValueError(f"Invalid node at index { i }, preceding node's outputs not detected in this node. This is not supported by Avrex. Please see readme.")

    gemm_flag = (last_node is not None) and ((node.op_type == "Add") and (last_node.op_type == "MatMul"))
    matmul_gemm_flag = (last_node is not None) and ((node.op_type != "Add") and (last_node.op_type == "MatMul"))

    # Create a Gemm layer
    if matmul_gemm_flag:
        tensors = [item for item in last_node.input if item in static_tensor_list]
        new_node = avrfunc.AvrexNode(op_type = "Gemm", static_tensors = tensors, output_name = node.output[0])
        to_create_list.append(new_node)

    # Dont want to create a Gemm layer early
    if node.op_type != "MatMul":
        if gemm_flag:
            tensors = [item for item in last_node.input if item in static_tensor_list] + [item for item in node.input if item in static_tensor_list] # Only grab static inputs
            new_node = avrfunc.AvrexNode(op_type = "Gemm", static_tensors = tensors, output_name = node.output[0])
            to_create_list.append(new_node)
        else:
            tensors = [item for item in node.input if item in static_tensor_list] # Only grab static inputs
            new_node = avrfunc.AvrexNode(op_type = node.op_type, static_tensors = tensors, output_name = node.output[0])
            to_create_list.append(new_node)

    # Check for transpositions
    if node.op_type in avrex_data_ops:
        weight_attributes = { a.name : onnx.helper.get_attribute_value(a) for a in node.attribute }
        transBAttr = weight_attributes.get("transB", 0)
        weight_transpose_flag = transBAttr == 0

    print(i, type(node).__name__, graph.node[i].op_type, transBAttr, weight_transpose_flag)

    # Upkeep
    last_node = node
    output_occurence_map[last_node.output[0]] = 0 # This nodes output now exists and is tracked

# Actually start creation process
model_header = HeaderFile(args.header_name)
initializer_dict = { init.name: init for init in model.graph.initializer }

global_weight_list = []
global_bias_list = []
weight_positions = [0] * len(to_create_list)
bias_positions = [0] * len(to_create_list)
alpha_positions = [0] * len(to_create_list)

total_weight_bytes = total_biases = 0

calibrater = calibration.AvrexCalibrater(model, args.calibration_data_path)
ranges = calibrater.find_ranges()
dp_array = calibrater.dp_from_ranges(ranges = ranges, avrex_nodes = to_create_list)

# Weight stuff
for i, node in enumerate(to_create_list):
    if node.op_type in avrex_data_ops:
        match node.op_type:
            case "Gemm":
                if not node.static_tensors[0] in initializer_dict:
                    raise ValueError("Tensor not in initializer dict, cannot read weights. NOTE: THIS IS NOT A USER ERROR, PLEASE REPORT BUG")

                weight_proto = initializer_dict[node.static_tensors[0]]

                # ADD QUANTIZATION PIPELINE EVENTUALLY
                proto_as_arr = nphelper.to_array(weight_proto).transpose() if weight_transpose_flag else nphelper.to_array(weight_proto)

                # Too much work to go hunt for alpha so it better be in there!
                if not quantization.validate_ternary_tensor(proto_as_arr):
                    raise ValueError("Found weight tensor not ternary in the network")
                    
                alpha = quantization.get_tensor_alpha(proto_as_arr)
                alpha_positions[i] = alpha
                normal_weight_arr = quantization.normalize_ternary_tensor(proto_as_arr)

                weight_list = quantization.pack_ternary(normal_weight_arr.flatten().tolist())
                global_weight_list.append(weight_list)

                weight_positions[i] = total_weight_bytes
                total_weight_bytes += len(weight_list)

                # If biases exist
                if len(node.static_tensors) > 1:
                    bias_proto = initializer_dict[node.static_tensors[1]]
                    flat_bias = nphelper.to_array(bias_proto).flatten()

                    bias_dp = dp_array[i]
                    bias_arr = quantization.biases_to_fixedp(flat_bias, bias_dp) if args.quantize_biases else flat_bias
                    global_bias_list.append(bias_arr.tolist())

                    bias_positions[i] = total_biases
                    total_biases += len(bias_arr)

                last_gemm = node

weight_arr = np.concatenate(global_weight_list)
bias_arr = np.concatenate(global_bias_list)

model_header.create_weights(weight_arr.astype(np.uint8))
model_header.create_biases(bias_arr.astype(int))

layer_types_list = []
layer_names_list = []

max_buffer_size = 0
cur_network_size = graph.input[0].type.tensor_type.shape.dim[-1].dim_value

for i, node in enumerate(to_create_list):
    layer_name = f"layer{ i }"
    layer_dp = dp_array[i]

    if node.op_type in avrex_data_ops:
        match node.op_type:
            case "Gemm":
                alpha = alpha_positions[i]
                alpha_terms, alpha_signs = quantization.generate_alpha_approx(alpha)
                model_header.create_alpha_approx_func(alpha_terms, alpha_signs, layer_index = str(i))
                alpha_func_ptr = model_header.create_alpha_approx_func_ptr(layer_index = str(i))

                # In a typical onnx format Gemm inputs are activations->weights->biases. Weights are first static tensor.
                weight_tensor = initializer_dict[node.static_tensors[0]]
                layer_size, prev_layer_size = (weight_tensor.dims[0], weight_tensor.dims[1]) if not weight_transpose_flag else (weight_tensor.dims[1], weight_tensor.dims[0])
                cur_network_size = layer_size

                max_buffer_size = max(max_buffer_size, layer_size, prev_layer_size)
                activate_bias = len(node.static_tensors) > 1

                bias_pointer = f"biases + { bias_positions[i] }" if activate_bias else "nullptr"
                weight_pointer = f"weights + { weight_positions[i] }"

                layer_types_list.append("DenseLayer")
                model_header.add_dense_layer(layer_name = layer_name, 
                                             layer_size = str(layer_size), 
                                             prev_layer_size = str(prev_layer_size),
                                             bias_pointer = bias_pointer,
                                             weight_pointer = weight_pointer,
                                             alpha_func_ptr = alpha_func_ptr,
                                             dp = str(layer_dp),
                                             use_bias = str(activate_bias).lower())
    elif node.op_type in avrex_activ_ops:
        activ_func = avrex_activ_ops_map[node.op_type]
        layer_types_list.append("ActivationFunc")
        model_header.add_activation_layer(layer_name = layer_name, activ_func = activ_func, layer_size = str(cur_network_size), dp = str(layer_dp)) # CHANGE FOR LUT CASES

    layer_names_list.append(layer_name)

model_header.create_network("network", buffer_size = str(max_buffer_size), layer_types = layer_types_list, layer_names = layer_names_list, input_dp = str(args.input_decimal_point))
model_header.wrap_model() # Gives includes and namespace etc...
print(model_header.contents)

file_path = os.path.join(args.header_output_path, args.header_name + ".hpp")
if not os.path.exists(args.header_output_path):
    raise ValueError("The specified header output path does not exist.")

file = open(file_path, "w")
file.write(model_header.contents)

"""
add inputs to tracked activations
add LUT DP case or supprot activations 
"""