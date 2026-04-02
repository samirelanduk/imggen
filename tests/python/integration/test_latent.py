from .base import IntegrationTestCase
import tempfile
import os
import shutil
import torch

class SaveBlankLatentTests(IntegrationTestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

    def test_save_blank_latent(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "save_blank_latent", "--path", os.path.join(self.temp_dir, "test.pt"), "--width", "800", "--height", "400"])
        self.assertEqual(proc.returncode, 0)
        with open(os.path.join(self.temp_dir, "test.pt"), "rb") as f:
            tensor = torch.load(f)
        self.assertEqual(tensor.shape, (1, 4, 50, 100))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.equal(tensor, torch.zeros(1, 4, 50, 100)))

    def test_path_is_required(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "save_blank_latent", "--width", "800", "--height", "400"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("the following arguments are required: --path", proc.stderr)

    def test_width_is_required(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "save_blank_latent", "--path", os.path.join(self.temp_dir, "test.pt"), "--height", "400"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("he following arguments are required: --width", proc.stderr)
    
    def test_width_must_be_integer(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "save_blank_latent", "--path", os.path.join(self.temp_dir, "test.pt"), "--width", "xxx", "--height", "400"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid int value: 'xxx'", proc.stderr)
    
    def test_height_is_required(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "save_blank_latent", "--path", os.path.join(self.temp_dir, "test.pt"), "--width", "800"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("the following arguments are required: --height", proc.stderr)
    
    def test_height_must_be_integer(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "save_blank_latent", "--path", os.path.join(self.temp_dir, "test.pt"), "--width", "800", "--height", "xxx"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid int value: 'xxx'", proc.stderr)

