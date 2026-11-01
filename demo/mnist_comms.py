import serial
import torchvision as tv
import time
import numpy as np
import onnxruntime as ort
import struct
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type = str, required = True)
parser.add_argument("--q_format", type = int, default = 14)
args = parser.parse_args()

transform_pipeline = tv.transforms.Compose([tv.transforms.Resize((14, 14)), tv.transforms.ToTensor()])
mnist_test = tv.datasets.MNIST(root = "avrex_tools/datasets/", train = False, transform = transform_pipeline, download = True)

session = ort.InferenceSession(f"avrex_tools/models/{ args.name }.onnx")

try:
    with serial.Serial("COM3", 500000, timeout = 5) as ser:
        print("Connect Success")
        time.sleep(2)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        to_fixed = lambda x: int(round(x * (2 ** args.q_format)))
        to_fixed_np = np.vectorize(to_fixed)

        total_time = 0
        num_correct = 0
        high_res_correct = 0
        for i in range(len(mnist_test)):
            image, label = mnist_test[i]
            x = image.numpy().reshape(1, 196).astype(np.float32)
            image = image.detach().cpu().numpy().flatten()

            out = session.run(None, { "Input" : x })[0]                
            high_res_correct += int(out.argmax() == label)

            fixed_image = to_fixed_np(image).astype(np.int16)

            bytes = fixed_image.tobytes()
            ser.write(bytes)

            guess = ser.read(5)
            if len(guess) == 5:
                predicted_digit, time = struct.unpack("<bi", guess)
                if predicted_digit == label:
                    num_correct += 1
                print(f"current accuracy { round(num_correct / (i + 1) * 100, 2) }%, { i + 1 }, time { time / 1000 }ms, guess { predicted_digit } || label { label }, High res { round(high_res_correct / (i + 1) * 100, 2) }%")
                total_time += time
            else:
                print("Timed out waiting for prediction.")

        print(f"model accuracy { num_correct / len(mnist_test) }, average time { (total_time / 1000) / len(mnist_test) }, high res accuracy { high_res_correct / len(mnist_test) }")

except serial.SerialException as exception:
    print(f"Serial failed { exception }")