import math

import torch.nn as nn
import torch


class BasicNeuralNetwork(nn.Linear):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)

    def calculate_value(self, input_tensor: torch.Tensor):
        self(input_tensor)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # This function is internally called from 'nn.Module' when we call the function
        # calculate_value on an input tensor. This function signifies forward pass.
        print(input.shape)
        print(self.weight.shape)
        print(self.bias.shape)

        # Perform forward pass which is going to calculate the following value -
        # y = (input) * (self.weight)^T + (self.bias)
        output_tensor = super().forward(input)
        print(output_tensor.shape)

        return output_tensor


basic_neural_network = BasicNeuralNetwork(
    input_shape=20,
    output_shape=30
)


basic_neural_network.calculate_value(torch.randn(128, 20))
print(basic_neural_network.parameters)

print('--------------------------------------------------------------------------------------------------------------')


class CustomBasicNeuralNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, device=None, dtype=None):
        super(CustomBasicNeuralNetwork, self).__init__()

        # The below two lines dictate the number of neurons in input and output features
        self.input_shape = input_shape
        self.output_shape = output_shape

        factory_argws = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(
            torch.empty((output_shape, input_shape), **factory_argws)
        )
        self.bias = nn.Parameter(
            torch.empty(output_shape, **factory_argws)
        )
        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        print('Input: ' + str(input.shape), 'Weights: ' + str(self.weight.shape), 'Bias: ' + str(self.bias.shape),
              sep='\n')

        output = input @ self.weight.t() + self.bias
        print('Output:', output.shape)

        return output

    # Code as picked up from nn.Linear
    # TODO - Need to understand this piece of code and the reason behind it.
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


custom_basic_neural_network = CustomBasicNeuralNetwork(
    input_shape=40,
    output_shape=50
)

custom_basic_neural_network(torch.randn(1024, 40))
print(custom_basic_neural_network.parameters)

print('--------------------------------------------------------------------------------------------------------------')
