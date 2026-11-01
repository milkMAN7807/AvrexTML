# AvrexTML
Neural Network Inference library and Toolchain that is designed for low end AVR chips. This is accomplished by supporting ternary NNs, where weights are constrained to -α, 0, and α per layer.
Avrex puts four weights into a byte, but activations are maintained as int16, meaning biases take two bytes. Everything is stored in flash so keep that in mind. Models rarely scale in RAM with depth but do with width, MAKE SLIM MODELS!
Please note this library is in very early development so there are plenty of bugs and areas that can be made better
- Graph validation is likely to be buggy, expect errors! and read them too.
- DP Shifting routine in firmware can be moved to compile time
- Buffer sizing is not optimized by the compiler yet which can save RAM

# How to use
The library ingests ONNX models in a specific format. NOTE THIS LIBRARY ONLY INGESTS PRE-QUANTIZED MODELS!!!
- all weights must be included in the onnx file
- all weights must be fp32 and all weight matrices must be ternarized
- Alpha for each dense layer must be inside the weight matrix
- The graph topology must be sequential (The compiler will throw a fit)
  - As of right now only MLPs are supported, so standard dense networks
- Please check the demo model maker script if you are unsure about anything
- Once you have your model header file ready, include it, feed it an int16 buffer with your data and run MyHeaderTitle::network.forward(myInt16Buffer, numberOfElements)
- Simply read whats in your buffer and voila!

# Benchmarks
All these benchmarks were run on the Atmega328p Arduino UNO R3
- Large model on MNIST digit dataset
  - 196->196->relu->10->softmax (40,582 paramters)
  - On device model accuracy: 97.14%
  - Average inference time: 49.33ms
  - ONNX model run on PC accuracy: 97.14%
- Small model on MNIST digit dataset
  - 196-20-relu-10-softmax
  - On device model accuracy: 93.05%
  - Average inference time: 6.72ms
  - ONNX model run on PC accuracy: 93.05%
