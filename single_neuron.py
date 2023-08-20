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


BasicNeuralNetwork(
    input_shape=20,
    output_shape=30
).calculate_value(torch.randn(128, 20))
