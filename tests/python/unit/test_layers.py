import torch
from unittest import TestCase
from unittest.mock import patch
from pydiffuser.layers import linear, group_norm

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


class GroupNormLayerTests(TestCase):

    def test_group_norm_layer(self):
        input = torch.tensor([[1., 2., 3., 4., 5., 6.], [2., 4., 8., 16., 32., 64.]])
        weights = torch.tensor([10., 20., 30., 40., 50., 60.])
        bias = torch.tensor([100., 200., 300., 400., 500., 600.])
        output = group_norm(weights, bias, input)
        self.assertTrue(
            torch.allclose(output,
            torch.tensor([
                [85.3615, 182.4338, 291.2169, 411.7108, 543.9154, 687.8309],
                [91.2266, 184.3003, 281.9915, 390.7649, 525.3966, 719.1333]
            ])
            )
        )
    
    def test_group_norm_layer_custom_group_size(self):
        input = torch.tensor([[1., 2., 3., 4., 5., 6.], [2., 4., 8., 16., 32., 64.]])
        weights = torch.tensor([10., 20., 30., 40., 50., 60.])
        bias = torch.tensor([100., 200., 300., 400., 500., 600.])
        output = group_norm(weights, bias, input, groups=2)
        self.assertTrue(
            torch.allclose(output,
            torch.tensor([
                [87.7526, 200.0000, 336.7421, 351.0106, 500.0000, 673.4841],
                [89.3096, 194.6548, 340.0891, 357.2382, 486.6370, 680.1783]
            ])
            )
        )