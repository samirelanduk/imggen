import torch
from unittest import TestCase
from unittest.mock import patch
from pydiffuser.latent import save_blank_latent

class TestLatent(TestCase):

    @patch("pydiffuser.latent.torch.save")
    def test_save_blank_latent(self, mock_save):
        save_blank_latent("test.pt", 800, 400)
        tensor_saved = mock_save.call_args[0][0]
        self.assertEqual(tensor_saved.shape, (1, 4, 50, 100))
        self.assertEqual(tensor_saved.dtype, torch.float32)
        self.assertTrue(torch.equal(tensor_saved, torch.zeros(1, 4, 50, 100)))