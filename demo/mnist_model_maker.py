import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as tv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, default = 250)
parser.add_argument("--batch_size", type = int, default = 50)
parser.add_argument("--learning_rate", type = float, default = 1e-3)
parser.add_argument("--name", type = str, required = True)
parser.add_argument("--q7_train", action = "store_true")
args = parser.parse_args()

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# Model building
# Ternarization implements 2016 TWN method
def as_ternary(input):
    abs_input = t.abs(input)
    delta = 0.7 * t.mean(abs_input[abs_input != 0])

    ternary_tensor = t.zeros_like(input)
    ternary_tensor[input > delta] = 1.0
    ternary_tensor[input < -delta] = -1.0
    return ternary_tensor

class STE(t.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return as_ternary(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(t.Tensor(out_features, in_features))
        self.bias = nn.Parameter(t.Tensor(out_features))
        self.alpha = nn.Parameter(t.tensor(1.0))

        nn.init.kaiming_uniform_(self.weight, a = 0.0)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            new_weight = self.alpha * STE.apply(self.weight)
        else:
            new_weight = self.alpha * as_ternary(self.weight)

        return nn.functional.linear(x, new_weight, self.bias)

    def get_ternary_weight(self):
        return self.alpha * as_ternary(self.weight)

# Model definition
model = nn.Sequential(
    TernaryLinear(196, 196),
    nn.ReLU(),
    TernaryLinear(196, 10),
    nn.Softmax()
).to(device)

optimizer = t.optim.Adam(model.parameters(), lr = args.learning_rate)
criterion = t.nn.CrossEntropyLoss()

if __name__ == "__main__":
    mnist_train = tv.datasets.MNIST(root = "avrex_tools/datasets/", train = True, download = True)
    mnist_test = tv.datasets.MNIST(root = "avrex_tools/datasets/", train = False, download = True)

    x_train_scaled = mnist_train.data.unsqueeze(1).float().to(device) / 255.0
    x_train = t.nn.functional.interpolate(x_train_scaled, size = (14, 14), mode = "bilinear", align_corners = False).flatten(start_dim = 1)
    y_train = mnist_train.targets.to(device)

    x_test_scaled = mnist_test.data.unsqueeze(1).float().to(device) / 255.0
    x_test = t.nn.functional.interpolate(x_test_scaled, size = (14, 14), mode = "bilinear", align_corners = False).flatten(start_dim = 1)
    y_test = mnist_test.targets.to(device)

    highest_accuracy = 0.0

    for i in range(args.epochs):
        model.train()

        permutation = t.randperm(x_train.size(0), device = device)
        for idx in range(0, x_train.size(0) - args.batch_size + 1, args.batch_size):
            batch_indices = permutation[idx:idx + args.batch_size]
            images, labels = x_train[batch_indices], y_train[batch_indices]
            
            optimizer.zero_grad(set_to_none = True)
        
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        # Test eval
        model.eval()
        correct = 0
        total = 0

        with t.inference_mode():
            for idx in range(0, x_test.size(0) - args.batch_size + 1, args.batch_size):
                images = x_test[idx:idx + args.batch_size]
                labels = y_test[idx:idx + args.batch_size]

                outputs = model(images)
                predictions = t.argmax(outputs, dim = 1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        cur_accuracy = correct / total
        if cur_accuracy > highest_accuracy:
            highest_accuracy = cur_accuracy
            t.save(model.state_dict(), f"avrex_tools/models/pytorch_checkpoints/{args.name}.pt")
            print(f"Saved new checkpoint, accuracy is { highest_accuracy }, epoch: { i }")

    best_state_dict = t.load(f"avrex_tools/models/pytorch_checkpoints/{args.name}.pt")
    model.load_state_dict(best_state_dict)

    if not args.q7_train:
        # Create calibration dataset
        calibration_inputs = []

        calibration_gpu = x_test[::25] 
        calibration_inputs = calibration_gpu.cpu().numpy()

        calibration_dict = { f"batch_{ i }" : np.expand_dims(arr, axis = 0).astype(np.float32) for i, arr in enumerate(calibration_inputs) }
        
        np.savez("avrex_tools/datasets/mnist_calibration.npz", **calibration_dict)

        # Export
        def clean_onnx(network):
            for name, module in network.named_children():
                if isinstance(module, TernaryLinear):
                    layer_weight = module.get_ternary_weight().detach().to('cpu')
                    standard_layer = nn.Linear(module.in_features, module.out_features, bias = True)

                    standard_layer.weight = nn.Parameter(layer_weight.clone())

                    layer_bias = module.bias.detach().to('cpu')
                    standard_layer.bias = nn.Parameter(layer_bias.clone())

                    setattr(network, name, standard_layer)
                else:
                    clean_onnx(module)

            return network

        uncompiled_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        clean_model = clean_onnx(uncompiled_model)

        dummy_in = x_test[0].unsqueeze(0).cpu() 
        t.onnx.export(model, dummy_in.cpu(), f"avrex_tools/models/{args.name}.onnx", dynamo = False, export_params = True, opset_version = 17, do_constant_folding = True, input_names = ["Input"], output_names = ["Output"])