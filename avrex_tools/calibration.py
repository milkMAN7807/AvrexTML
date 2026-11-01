import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader
import numpy as np
from avrex_functions import avrex_lut_nodes, avrex_lut_dp_map

nodes_to_calibrate = ["Add", "Gemm", "Relu"]

class NpzDataReader(CalibrationDataReader):
    def __init__(self, dataset_npz, input_node):
        self.dataset_iter = iter(dataset_npz.values())
        self.input_node = input_node

    def get_next(self) -> dict:
        try:
            return { self.input_node : next(self.dataset_iter).astype(np.float32) }
        except StopIteration:
            return None

class AvrexCalibrater:
    # Data set needs to be a .npz
    def __init__(self, model, dataset_path):
        self.model = model
        graph = model.graph
        
        inputs = [i.name for i in graph.input if i.name not in [t.name for t in graph.initializer]]
        if len(inputs) > 1:
            raise ValueError("More than one input detected in the graph")
        
        input_name = inputs[0] # Guaranteed one input
        self.input_name = input_name
        
        marked_input = onnx.helper.make_tensor_value_info(name = input_name, elem_type = onnx.TensorProto.FLOAT, shape = None)
        self.model.graph.output.append(marked_input)

        # In order to inspect each node
        model_outputs = { output.name for output in graph.output }
        
        for node in graph.node:
            if node.op_type in nodes_to_calibrate:    
                output_name = node.output[0] # Guaranteed one output
                if output_name not in model_outputs:
                    marked_output = onnx.helper.make_tensor_value_info(name = output_name, elem_type = onnx.TensorProto.FLOAT, shape = None)
                    self.model.graph.output.append(marked_output)
                    model_outputs.add(output_name)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL # It shouldve already passed avrex inspection so no need

        self.session = ort.InferenceSession(model.SerializeToString(), options, providers = ["CPUExecutionProvider"])
        self.data_reader = NpzDataReader(dataset_npz = np.load(dataset_path), input_node = self.session.get_inputs()[0].name)
        
    def find_ranges(self):
        output_names = [output.name for output in self.session.get_outputs()]
        node_activation_dict = {}
        range_dict = {}

        while True:
            input_data = self.data_reader.get_next()
            if input_data is None:
                break

            activations = self.session.run(output_names, input_data)
            # activations[i] = NN(output_name[i])

            for i, activation in enumerate(activations):
                n_activation = activation.flatten()
                output_name = output_names[i]
                inserted_arr = np.concatenate((node_activation_dict[output_name], n_activation)) if output_name in node_activation_dict else n_activation
                node_activation_dict[output_name] = inserted_arr
                
        def filter_and_append(output_name):
            activations = node_activation_dict[output_name]
            d1, d2 = np.percentile(activations, [1.5, 98.5]) # No way to effectively remove outliers besides intuition. Large values do ruin the network, yet large vals are needed because they get redistributed within layers
            #v = np.max(np.abs(activations[(activations >= d1) & (activations <= d2)]))
            v = np.max(np.abs(activations))
            range_dict[output_name] = v

        # Process 
        filter_and_append(self.input_name)
        for node in self.model.graph.node:
            if node.op_type in nodes_to_calibrate:
                output_name = node.output[0]
                if output_name in node_activation_dict:
                    filter_and_append(output_name)

        return range_dict
    
    def dp_from_ranges(self, ranges: dict, avrex_nodes):
        range_to_dp = lambda r: 14 - int(np.ceil(np.log2(r))) # 16th bit is sign bit, absolue hail mary with the + 1
        
        out_dps = [range_to_dp(ranges[self.input_name])]
        for node in avrex_nodes:
            if node.op_type in nodes_to_calibrate:
                out_dps.append(range_to_dp(ranges[node.output_name]))
            elif node.op_type in avrex_lut_nodes:
                out_dps.append(avrex_lut_dp_map[node.op_type])
            else:
                raise ValueError("Node not in calibrated nodes nor LUT nodes, could not assign proper DP")
            
        new_dps = out_dps.copy()
        for i in range(1, len(out_dps)):
            new_dps[i] = min(out_dps[i - 1], out_dps[i]) # dp is fractional bits
        
        return new_dps[1:]