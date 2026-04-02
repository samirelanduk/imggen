from .base import IntegrationTestCase
import tempfile
import os
import shutil
import json

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