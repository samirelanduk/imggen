import torch
from unittest import TestCase
from unittest.mock import patch
from pydiffuser.layers import linear

class LinearLayerTests(TestCase):

    def test_linear_layer_vector_input(self):
        input = torch.tensor([10, 20, 30, 40])
        weights = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        bias = torch.tensor([12, 13, 14])
        output = linear(weights, bias, input)
        self.assertEqual(output.tolist(), [312, 713, 1114])
    
    def test_linear_layer_matrix_input(self):
        input = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]])
        weights = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        bias = torch.tensor([12, 13, 14])
        output = linear(weights, bias, input)
        self.assertEqual(output.tolist(), [[312, 713, 1114], [712, 1753, 2794]])