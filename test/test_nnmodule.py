#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/12/8 20:41
# @Author : Tao Zhang
# @Email: 2637050370@qq.com
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Example usage:
input_size = 10
hidden_size = 20
output_size = 5

# Instantiate the network
simple_net = SimpleNet(input_size, hidden_size, output_size)

# Print the network architecture
print(simple_net)
