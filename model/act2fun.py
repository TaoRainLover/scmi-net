import torch
import math


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# swish 激活函数
def swish(x):
    return x * torch.sigmoid(x)


# 激活函数集
ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
