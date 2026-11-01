import textwrap
from string import Template

file_template = """#pragma once
#include <avrex/Tensor.hpp>
#include <avrex/TensorMath.hpp>
#include <avrex/Activations.hpp>
#include <avrex/DataStructures.hpp>
#include <avrex/DenseLayer.hpp>
#include <avrex/LookUpTable.hpp>
#include <avrex/LookUpTableNumerics.hpp>
#include <avrex/Memory.hpp>
#include <avrex/Network.hpp>
#include <avrex/Numerics.hpp>

"""

def list_as_string(vals):
    created_list = ""
    for i in range(len(vals)):
        if i == len(vals) - 1:
            created_list += str(vals[i])
        else:
            created_list += str(vals[i]) + ", "

    return created_list

class HeaderFile:
    def __init__(self, header_title):
        self.contents = ""
        self.header_title = header_title

    def wrap_model(self):
        indented_contents = textwrap.indent(self.contents, "    ")
        self.contents = f"namespace { self.header_title } {{\n" + indented_contents + "}"
        self.contents = file_template + self.contents

    def create_network(self, network_name, buffer_size, layer_types, layer_names, input_dp):
        types_string = list_as_string(layer_types)
        names_string = list_as_string(layer_names)

        network_template = Template("Network<${buffer_size}, ${layer_type_string}> ${network_name}(${layer_name_string}, ${input_dp});\n")
        network_value = network_template.substitute(buffer_size = buffer_size, layer_type_string = types_string, network_name = network_name, layer_name_string = names_string, input_dp = input_dp)

        self.contents += network_value

    def add_dense_layer(self, layer_name, layer_size, prev_layer_size, bias_pointer, weight_pointer, alpha_func_ptr, dp, use_bias):
        layer_template = Template("DenseLayer ${layer_name}(${layer_size}, ${prev_layer_size}, ${bias_pointer}, ${weight_pointer}, ${alpha_func_ptr}, ${dp}, ${use_bias});\n")
        layer_value = layer_template.substitute(layer_name = layer_name, 
                                                layer_size = layer_size, 
                                                prev_layer_size = prev_layer_size, 
                                                bias_pointer = bias_pointer, 
                                                weight_pointer = weight_pointer, 
                                                alpha_func_ptr = alpha_func_ptr,
                                                dp = dp, 
                                                use_bias = use_bias)
        self.contents += layer_value

    def add_activation_layer(self, layer_name, activ_func, layer_size, dp):
        layer_template = Template("ActivationFunc ${layer_name}(&ActivationLib::${activ_func}, ${layer_size}, ${dp});\n")
        layer_value = layer_template.substitute(layer_name = layer_name, activ_func = activ_func, layer_size = layer_size, dp = dp)
        self.contents += layer_value

    def create_weights(self, weight_arr):
        weight_string = list_as_string(weight_arr)
        arr_template = Template("const TernaryPack weights[] TRY_FLASH = { ${vals} };\n")

        arr_value = arr_template.substitute(vals = weight_string)
        self.contents += arr_value
        
    def create_biases(self, bias_arr):
        bias_string = list_as_string(bias_arr)
        arr_template = Template("const int16_t biases[] TRY_FLASH = { ${vals} };\n")

        arr_value = arr_template.substitute(vals = bias_string)
        self.contents += arr_value
        
    def create_alpha_approx_func(self, terms, signs, layer_index):
        value = f"inline void multiplyAlpha{ layer_index }(int16_t* bufferA, int16_t* bufferB, uint16_t size)"
        value += "{ for (uint16_t i = 0; i < size; i++) bufferB[i] = "

        def gen_shift_string(log_scalar):
            if log_scalar > 0:
                return f"FixedPointMath::satSignedLeftShift(bufferA[i], { log_scalar }) "
            elif log_scalar == 0:
                return "bufferA[i] "
            else:
                return f"(bufferA[i] >> { -log_scalar }) "
        
        for i in range(len(terms)):
            value += "- " if signs[i] < 0 else ("+ " if i > 0 else "") 
            value += gen_shift_string(terms[i])
                
        value += "; }\n"
        self.contents += value

    def create_alpha_approx_func_ptr(self, layer_index):
        self.contents += f"void (*const multiplyAlphaPtr{ layer_index })(int16_t*, int16_t*, uint16_t) = &multiplyAlpha{ layer_index };\n"
        return f"multiplyAlphaPtr{ layer_index }" # Give name for logging purposes
            
        
