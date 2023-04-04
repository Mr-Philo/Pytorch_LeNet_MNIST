import torch
from torch import nn
import torch.nn.functional as F
import math
 
 
class Conv2Linear(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        self.unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), padding=padding, stride=stride)
        self.linear = nn.Linear(in_channels * kernel_size * kernel_size, out_channels, bias=bias)

            
    def add_weight_and_bias(self, weight, bias):
        self.linear.weight = nn.Parameter(weight)
        self.linear.bias = nn.Parameter(bias)
        
    def forward(self, torch_conv):     
        batch_size = torch_conv.shape[0]
        assert torch_conv.shape[1] == self.in_channels
        input_h, input_w = torch_conv.shape[2:4]
        out_h = math.floor((input_h + 2*self.padding - self.kernel_size) / self.stride + 1)
        out_w = math.floor((input_w + 2*self.padding - self.kernel_size) / self.stride + 1)
        
        # print(f"before unfold: {torch_conv.shape}")  # torch.Size([4, 3, 5, 5])
        torch_conv = self.unfold(torch_conv)
        # print(f"after unfold: {torch_conv.shape}")   # torch.Size([4, 27, 9])
        torch_conv = torch_conv.permute(0,2,1).contiguous()
        # print(f"after permute: {torch_conv.shape}")  # torch.Size([4, 9, 27])
        torch_conv = self.linear(torch_conv)
        torch_conv = torch_conv.reshape(batch_size, out_h, out_w, self.out_channels).permute(0,3,1,2).contiguous()
        return torch_conv
        
        
        
def mim2col_conv_conv(input, kernel, stride=1, padding=0, bias=0):
    # directly compute convolution using im2col, compatible with F.conv2d
    if padding > 0:
        input = F.pad(input, (padding,padding,padding,padding))
    batch_size = input.shape[0]
    input_h, input_w = input.shape[2:4]
    kernel_h, kernel_w = kernel.shape[2:4]
    out_channels, in_channels = kernel.shape[0:2]
    output_h = math.floor((input_h - kernel_h) / stride + 1)
    output_w = math.floor((input_w - kernel_w) / stride + 1)
    
    # Convolution is equivalent with Unfold + Matritorch_conv Multiplication + Fold (or view to output shape)
    unfold = nn.Unfold(kernel_size=(kernel_h, kernel_w), stride=stride)
    input_vector = unfold(input)
    
    kernel_vector = kernel.reshape(kernel.shape[0], -1).T
    output = (input_vector.permute(0,2,1).contiguous() @ kernel_vector ) + bias
    output = output.reshape(batch_size, output_h, output_w, out_channels).permute(0,3,1,2).contiguous()    
    
    # do not write like this
    # output = output.reshape(batch_size, out_channels, output_h, output_w)
    
    return output
 
 
if __name__ == "__main__":
    
    print(">>> Unit test for Conv2Linear")
    batch_size = 4
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    input = torch.rand(batch_size, in_channels ,5,5)
    kernel = torch.rand(out_channels, in_channels, 3,3)
    bias = torch.rand(out_channels)
    
    stride = 2
    padding = 1
    
    '''torch.nn.Cov2d, using specific weight and bias. Weight is a 4D tensor, representing the convolution kernel'''
    torch_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    torch_conv.weight = nn.Parameter(kernel)
    torch_conv.bias = nn.Parameter(bias)
    
    '''Conv2Linear, using specific weight and bias. Weight is a 2D tensor, flattened convolution kernel'''
    # print(f"before reshape: {kernel.shape}")  # torch.Size([16, 3, 3, 3])
    kernel_vector = nn.Parameter(kernel.reshape(kernel.shape[0], -1))
    # print(f"after reshape: {kernel_vector.shape}")   # torch.Size([16, 27])
    im2col_conv = Conv2Linear(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
    im2col_conv.add_weight_and_bias(weight=kernel_vector, bias=bias)
    
    torch_conv_out = torch_conv(input)
    im2col_conv_out = im2col_conv(input)
    print(f"torch_conv_out.shape: {torch_conv_out.shape}")
    print(f"im2col_conv_out.shape: {im2col_conv_out.shape}")
    assert torch.allclose(torch_conv_out, im2col_conv_out)
    print("Success!")
