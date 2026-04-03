from .base import IntegrationTestCase
import tempfile
import os
import shutil
import json
import torch

class ClipTokenizeTests(IntegrationTestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_clip_tokenize_short_prompt(self):
        # Run command
        prompt = "A beautiful sunset over the ocean."
        proc = self.run_command(["python", "-m", "pydiffuser", "clip_tokenize", "--text", prompt, "--clip_tokenizer", "assets/clip_tokenizer", "--tokens", os.path.join(self.temp_dir, "tokens.json"), "--mappings", os.path.join(self.temp_dir, "mappings.csv")])
        self.assertEqual(proc.returncode, 0)

        # Tokens saved
        with open(os.path.join(self.temp_dir, "tokens.json"), "r") as f:
            tokens = json.load(f)
        self.assertEqual(tokens, [[49406, 320, 1215, 3424, 962, 518, 4918, 269] + [49407] * 69])

        # Mappings saved
        with open(os.path.join(self.temp_dir, "mappings.csv")) as f:
            lines = f.read().splitlines()
        self.assertEqual(lines, [
            "<|startoftext|>,49406",
            "a</w>,320",
            "beautiful</w>,1215",
            "sunset</w>,3424",
            "over</w>,962",
            "the</w>,518",
            "ocean</w>,4918",
            ".</w>,269",
        ] + ["<|endoftext|>,49407"] * 69)
    
    def test_clip_tokenize_long_prompt(self):
        # Run command
        prompt = "A beautiful sunset over the ocean. The sky is clear and the ocean is calm. There are dolphins leaping, whales breaching, and birds flying overhead. It's the sort of thing you might see on a postcard. " * 3
        proc = self.run_command(["python", "-m", "pydiffuser", "clip_tokenize", "--text", prompt, "--clip_tokenizer", "assets/clip_tokenizer", "--tokens", os.path.join(self.temp_dir, "tokens.json"), "--mappings", os.path.join(self.temp_dir, "mappings.csv")])
        self.assertEqual(proc.returncode, 0)

        # Tokens saved
        with open(os.path.join(self.temp_dir, "tokens.json"), "r") as f:
            tokens = json.load(f)
        self.assertEqual(tokens, [
            [49406, 320, 1215, 3424, 962, 518, 4918, 269, 518, 2390, 533, 3143, 537, 518, 4918, 533, 7011, 269, 997, 631, 14002, 534, 18456, 267, 17722, 887, 12864, 267, 537, 4337, 4610, 20321, 269, 585, 568, 518, 8450, 539, 946, 592, 2727, 862, 525, 320, 14785, 269, 320, 1215, 3424, 962, 518, 4918, 269, 518, 2390, 533, 3143, 537, 518, 4918, 533, 7011, 269, 997, 631, 14002, 534, 18456, 267, 17722, 887, 12864, 267, 537, 4337, 4610, 49407],
            [49406, 20321, 269, 585, 568, 518, 8450, 539, 946, 592, 2727, 862, 525, 320, 14785, 269, 320, 1215, 3424, 962, 518, 4918, 269, 518, 2390, 533, 3143, 537, 518, 4918, 533, 7011, 269, 997, 631, 14002, 534, 18456, 267, 17722, 887, 12864, 267, 537, 4337, 4610, 20321, 269, 585, 568, 518, 8450, 539, 946, 592, 2727, 862, 525, 320, 14785, 269, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407]
        ])

        # Mappings saved
        with open(os.path.join(self.temp_dir, "mappings.csv")) as f:
            lines = f.read().splitlines()
        self.assertEqual(lines, [
            "<|startoftext|>,49406",
            "a</w>,320",
            "beautiful</w>,1215",
            "sunset</w>,3424",
            "over</w>,962",
            "the</w>,518",
            "ocean</w>,4918",
            ".</w>,269",
            "the</w>,518",
            "sky</w>,2390",
            "is</w>,533",
            "clear</w>,3143",
            "and</w>,537",
            "the</w>,518",
            "ocean</w>,4918",
            "is</w>,533",
            "calm</w>,7011",
            ".</w>,269",
            "there</w>,997",
            "are</w>,631",
            "dolphins</w>,14002",
            "le,534",
            "aping</w>,18456",
            "\",</w>\",267",
            "whales</w>,17722",
            "bre,887",
            "aching</w>,12864",
            "\",</w>\",267",
            "and</w>,537",
            "birds</w>,4337",
            "flying</w>,4610",
            "overhead</w>,20321",
            ".</w>,269",
            "it</w>,585",
            "'s</w>,568",
            "the</w>,518",
            "sort</w>,8450",
            "of</w>,539",
            "thing</w>,946",
            "you</w>,592",
            "might</w>,2727",
            "see</w>,862",
            "on</w>,525",
            "a</w>,320",
            "postcard</w>,14785",
            ".</w>,269",
            "a</w>,320",
            "beautiful</w>,1215",
            "sunset</w>,3424",
            "over</w>,962",
            "the</w>,518",
            "ocean</w>,4918",
            ".</w>,269",
            "the</w>,518",
            "sky</w>,2390",
            "is</w>,533",
            "clear</w>,3143",
            "and</w>,537",
            "the</w>,518",
            "ocean</w>,4918",
            "is</w>,533",
            "calm</w>,7011",
            ".</w>,269",
            "there</w>,997",
            "are</w>,631",
            "dolphins</w>,14002",
            "le,534",
            "aping</w>,18456",
            "\",</w>\",267",
            "whales</w>,17722",
            "bre,887",
            "aching</w>,12864",
            "\",</w>\",267",
            "and</w>,537",
            "birds</w>,4337",
            "flying</w>,4610",
            "<|endoftext|>,49407",
            "",
            "<|startoftext|>,49406",
            "overhead</w>,20321",
            ".</w>,269",
            "it</w>,585",
            "'s</w>,568",
            "the</w>,518",
            "sort</w>,8450",
            "of</w>,539",
            "thing</w>,946",
            "you</w>,592",
            "might</w>,2727",
            "see</w>,862",
            "on</w>,525",
            "a</w>,320",
            "postcard</w>,14785",
            ".</w>,269",
            "a</w>,320",
            "beautiful</w>,1215",
            "sunset</w>,3424",
            "over</w>,962",
            "the</w>,518",
            "ocean</w>,4918",
            ".</w>,269",
            "the</w>,518",
            "sky</w>,2390",
            "is</w>,533",
            "clear</w>,3143",
            "and</w>,537",
            "the</w>,518",
            "ocean</w>,4918",
            "is</w>,533",
            "calm</w>,7011",
            ".</w>,269",
            "there</w>,997",
            "are</w>,631",
            "dolphins</w>,14002",
            "le,534",
            "aping</w>,18456",
            "\",</w>\",267",
            "whales</w>,17722",
            "bre,887",
            "aching</w>,12864",
            "\",</w>\",267",
            "and</w>,537",
            "birds</w>,4337",
            "flying</w>,4610",
            "overhead</w>,20321",
            ".</w>,269",
            "it</w>,585",
            "'s</w>,568",
            "the</w>,518",
            "sort</w>,8450",
            "of</w>,539",
            "thing</w>,946",
            "you</w>,592",
            "might</w>,2727",
            "see</w>,862",
            "on</w>,525",
            "a</w>,320",
            "postcard</w>,14785",
            ".</w>,269",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407",
            "<|endoftext|>,49407"
        ])
    
    def test_prompt_is_required(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "clip_tokenize", "--clip_tokenizer", "assets/clip_tokenizer", "--tokens", os.path.join(self.temp_dir, "tokens.json"), "--mappings", os.path.join(self.temp_dir, "mappings.csv")])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("the following arguments are required: --text", proc.stderr)
    
    def test_clip_tokenizer_is_required(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "clip_tokenize", "--text", "A beautiful sunset over the ocean.", "--tokens", os.path.join(self.temp_dir, "tokens.json"), "--mappings", os.path.join(self.temp_dir, "mappings.csv")])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("the following arguments are required: --clip_tokenizer", proc.stderr)


class ClipEmbedTests(IntegrationTestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_clip_embed(self):
        # Run command
        proc = self.run_command(["python", "-m", "pydiffuser", "clip_embed", "--tokens", os.path.join("tests", "data", "tokens.json"), "--model", os.path.join("tests", "data", "clip_embedding_model.safetensors"), "--output", os.path.join(self.temp_dir, "embeddings.pt")])
        self.assertEqual(proc.returncode, 0)

        # Embeddings saved
        with open(os.path.join(self.temp_dir, "embeddings.pt"), "rb") as f:
            embeddings = torch.load(f)
        self.assertEqual(embeddings.shape, (3, 77, 12))
        self.assertEqual(round(embeddings[0, 0, 0].item(), 3), 0.385)
        self.assertEqual(round(embeddings[0, 0, 11].item(), 3), 0.036)
        self.assertEqual(round(embeddings[0, 76, 0].item(), 3), -2.082)
        self.assertEqual(round(embeddings[0, 76, 11].item(), 3), -0.533)
        self.assertEqual(round(embeddings[1, 0, 0].item(), 3), -0.001)
        self.assertEqual(round(embeddings[1, 0, 11].item(), 3), 1.848)
        self.assertEqual(round(embeddings[1, 76, 0].item(), 3), -1.012)
        self.assertEqual(round(embeddings[1, 76, 11].item(), 3), 0.025)
        self.assertEqual(round(embeddings[2, 0, 0].item(), 3), 0.534)
        self.assertEqual(round(embeddings[2, 0, 11].item(), 3), 1.369)
        self.assertEqual(round(embeddings[2, 76, 0].item(), 3), -1.012)
        self.assertEqual(round(embeddings[2, 76, 11].item(), 3), 0.025)
    
    def test_tokens_file_is_required(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "clip_embed", "--model", os.path.join("tests", "data", "clip_embedding_model.safetensors"), "--output", os.path.join(self.temp_dir, "embeddings.pt")])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("the following arguments are required: --tokens", proc.stderr)
    
    def test_model_file_is_required(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "clip_embed", "--tokens", os.path.join("tests", "data", "tokens.json"), "--output", os.path.join(self.temp_dir, "embeddings.pt")])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("the following arguments are required: --model", proc.stderr)


class ClipEncodeTests(IntegrationTestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def test_clip_encode(self):
        # Run command
        proc = self.run_command(["python", "-m", "pydiffuser", "clip_encode", "--embedding", os.path.join("tests", "data", "embedding.pt"), "--model", os.path.join("tests", "data", "clip_encode_model.safetensors"), "--output", os.path.join(self.temp_dir, "conditioning.pt")])
        self.assertEqual(proc.returncode, 0)

        # Encoding saved
        with open(os.path.join(self.temp_dir, "conditioning.pt"), "rb") as f:
            conditioning = torch.load(f)
        self.assertEqual(conditioning.shape, (3, 77, 12))
        self.assertEqual(round(conditioning[0, 0, 0].item(), 3), 0.698)
        self.assertEqual(round(conditioning[0, 0, 11].item(), 3), -0.011)
        self.assertEqual(round(conditioning[0, 76, 0].item(), 3), 0.556)
        self.assertEqual(round(conditioning[0, 76, 11].item(), 3), -0.601)
        self.assertEqual(round(conditioning[1, 0, 0].item(), 3), 0.541)
        self.assertEqual(round(conditioning[1, 0, 11].item(), 3), -0.075)
        self.assertEqual(round(conditioning[1, 76, 0].item(), 3), 0.938)
        self.assertEqual(round(conditioning[1, 76, 11].item(), 3), 0.45)
        self.assertEqual(round(conditioning[2, 0, 0].item(), 3), -0.292)
        self.assertEqual(round(conditioning[2, 0, 11].item(), 3), 0.289)
        self.assertEqual(round(conditioning[2, 76, 0].item(), 3), 0.575)
        self.assertEqual(round(conditioning[2, 76, 11].item(), 3), -0.85)
    
    def test_embedding_file_is_required(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "clip_encode", "--model", os.path.join("tests", "data", "clip_encode_model.safetensors"), "--output", os.path.join(self.temp_dir, "conditioning.pt")])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("the following arguments are required: --embedding", proc.stderr)
    
    def test_model_file_is_required(self):
        proc = self.run_command(["python", "-m", "pydiffuser", "clip_encode", "--embedding", os.path.join("tests", "data", "embedding.pt"), "--output", os.path.join(self.temp_dir, "conditioning.pt")])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("the following arguments are required: --model", proc.stderr)