import os
import torch
from unittest import TestCase
from unittest.mock import patch, mock_open, call, MagicMock
from transformers import CLIPTokenizer
from pydiffuser.clip import (
    tokenize,
    embed,
    _text_to_tokens,
    _break_up_tokens,
    _create_token_string_mapping,
    _create_token_embedding,
    _create_position_embedding,
)

class TokenizeTests(TestCase):

    @patch("pydiffuser.clip.CLIPTokenizer.from_pretrained")
    @patch("pydiffuser.clip._text_to_tokens")
    @patch("pydiffuser.clip._break_up_tokens")
    @patch("pydiffuser.clip._create_token_string_mapping")
    @patch("pydiffuser.clip.open", new_callable=mock_open)
    @patch("pydiffuser.clip.json.dump")
    @patch("pydiffuser.clip.csv.writer")
    def test_tokenize(self, mock_writer, mock_dump, mock_open, mock_create_mapping, mock_break_up, mock_to_tokens, mock_from_pretrained):
        mock_create_mapping.return_value = [[("the", 599)], [("big", 1915), ("one", 1980)], [("is", 595), ("coming", 14916)]]
        tokenize("A photo of a cat.", "assets/clip_tokenizer")
        mock_from_pretrained.assert_called_once_with("assets/clip_tokenizer")
        mock_to_tokens.assert_called_once_with("A photo of a cat.", mock_from_pretrained.return_value)
        mock_break_up.assert_called_once_with(mock_to_tokens.return_value, mock_from_pretrained.return_value)
        mock_create_mapping.assert_called_once_with(mock_break_up.return_value, mock_from_pretrained.return_value)
        mock_open.assert_any_call("tokens.json", "w")
        mock_open.assert_any_call("mappings.csv", "w", newline="")
        mock_dump.assert_called_once_with(mock_break_up.return_value, mock_open.return_value, indent=4)
        mock_writer.assert_called_once_with(mock_open.return_value)
        self.assertEqual(mock_writer.return_value.writerows.call_args_list, [call([("the", 599)]), call([("big", 1915), ("one", 1980)]), call([("is", 595), ("coming", 14916)])])
        self.assertEqual(mock_writer.return_value.writerow.call_args_list, [call([]), call([])])
    
    @patch("pydiffuser.clip.CLIPTokenizer.from_pretrained")
    @patch("pydiffuser.clip._text_to_tokens")
    @patch("pydiffuser.clip._break_up_tokens")
    @patch("pydiffuser.clip._create_token_string_mapping")
    @patch("pydiffuser.clip.open", new_callable=mock_open)
    @patch("pydiffuser.clip.json.dump")
    @patch("pydiffuser.clip.csv.writer")
    def test_tokenize_with_custom_paths(self, mock_writer, mock_dump, mock_open, mock_create_mapping, mock_break_up, mock_to_tokens, mock_from_pretrained):
        mock_create_mapping.return_value = [[("the", 599)], [("big", 1915), ("one", 1980)], [("is", 595), ("coming", 14916)]]
        tokenize("A photo of a cat.", "assets/clip_tokenizer", "custom_tokens.json", "custom_mappings.csv")
        mock_from_pretrained.assert_called_once_with("assets/clip_tokenizer")
        mock_to_tokens.assert_called_once_with("A photo of a cat.", mock_from_pretrained.return_value)
        mock_break_up.assert_called_once_with(mock_to_tokens.return_value, mock_from_pretrained.return_value)
        mock_create_mapping.assert_called_once_with(mock_break_up.return_value, mock_from_pretrained.return_value)
        mock_open.assert_any_call("custom_tokens.json", "w")
        mock_open.assert_any_call("custom_mappings.csv", "w", newline="")
        mock_dump.assert_called_once_with(mock_break_up.return_value, mock_open.return_value, indent=4)
        mock_writer.assert_called_once_with(mock_open.return_value)
        self.assertEqual(mock_writer.return_value.writerows.call_args_list, [call([("the", 599)]), call([("big", 1915), ("one", 1980)]), call([("is", 595), ("coming", 14916)])])
        self.assertEqual(mock_writer.return_value.writerow.call_args_list, [call([]), call([])])


class EmbedTests(TestCase):

    @patch("pydiffuser.clip.open", new_callable=mock_open)
    @patch("pydiffuser.clip.json.load")
    @patch("pydiffuser.clip.safetensors.safe_open")
    @patch("pydiffuser.clip._create_token_embedding")
    @patch("pydiffuser.clip._create_position_embedding")
    @patch("pydiffuser.clip.torch.save")
    def test_embed(self, mock_save, mock_pos_emb, mock_token_emb, mock_safe_open, mock_load, mock_open):
        mock_load.return_value = [[1, 2, 3], [4, 5, 6]]
        mock_token_emb.return_value = [10, 20]
        mock_pos_emb.return_value = [30, 40]
        embed("tokens.json", "model.safetensors", "output.pt")
        mock_open.assert_called_once_with("tokens.json")
        mock_load.assert_called_once_with(mock_open.return_value)
        mock_safe_open.assert_called_once_with("model.safetensors", framework="pt", device="cpu")
        safe_open_handle = mock_safe_open.return_value.__enter__.return_value
        self.assertEqual(mock_token_emb.call_args[0][0], safe_open_handle)
        self.assertTrue(torch.equal(mock_token_emb.call_args[0][1], torch.tensor([[1, 2, 3], [4, 5, 6]])))
        self.assertEqual(mock_pos_emb.call_args[0][0], safe_open_handle)
        self.assertTrue(torch.equal(mock_pos_emb.call_args[0][1], torch.tensor([[1, 2, 3], [4, 5, 6]])))
        mock_save.assert_called_once_with([10, 20, 30, 40], "output.pt")


class TextToTokensTests(TestCase):

    def test_text_to_tokens(self):
        prompt = "A photo of a cat."
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(os.getcwd(), "assets", "clip_tokenizer"))
        tokens = _text_to_tokens(prompt, tokenizer)
        self.assertEqual(tokens, [320, 1125, 539, 320, 2368, 269])



class BreakUpTokensTests(TestCase):

    def test_short_list(self):
        tokens = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(os.getcwd(), "assets", "clip_tokenizer"))
        result = _break_up_tokens(tokens, tokenizer, max_length=20)
        self.assertEqual(result, [[49406, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407]])
    

    def test_list_of_length_max_length(self):
        tokens = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(os.getcwd(), "assets", "clip_tokenizer"))
        result = _break_up_tokens(tokens, tokenizer, max_length=13)
        self.assertEqual(result, [[49406, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 49407]])
    

    def test_list_longer_than_max_length(self):
        tokens = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(os.getcwd(), "assets", "clip_tokenizer"))
        result = _break_up_tokens(tokens, tokenizer, max_length=8)
        self.assertEqual(result, [[49406, 1, 2, 3, 4, 5, 10, 49407], [49406, 11, 12, 13, 14, 15, 49407, 49407]])


class CreateTokenStringMappingTests(TestCase):

    def test_map_tokens_to_strings_single_list(self):
        tokens = [[599, 1915, 1980]]
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(os.getcwd(), "assets", "clip_tokenizer"))
        result = _create_token_string_mapping(tokens, tokenizer)
        self.assertEqual(result, [[("the", 599), ("big", 1915), ("one", 1980)]])
    
    def test_map_tokens_to_strings_multiple_lists(self):
        tokens = [[599, 1915, 1980], [595, 14916]]
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(os.getcwd(), "assets", "clip_tokenizer"))
        result = _create_token_string_mapping(tokens, tokenizer)
        self.assertEqual(result, [[("the", 599), ("big", 1915), ("one", 1980)], [("is", 595), ("coming", 14916)]])


class CreateTokenEmbeddingTests(TestCase):

    def test_creates_token_embedding(self):
        tensors = MagicMock()
        tensors.keys.return_value = ["1", "2", "xxx.text_model.token_embedding.weight", "3"]
        tensors.get_tensor.return_value = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        tokens = torch.tensor([[0, 2, 1]])
        result = _create_token_embedding(tensors, tokens)
        expected = torch.tensor([[[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]]])
        self.assertTrue(torch.equal(result, expected))

    def test_raises_if_no_token_embedding(self):
        tensors = MagicMock()
        tensors.keys.return_value = ["something_else.weight"]
        tokens = torch.tensor([[0, 1]])
        with self.assertRaises(ValueError):
            _create_token_embedding(tensors, tokens)


class CreatePositionEmbeddingTests(TestCase):

    def test_creates_position_embedding(self):
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        tensors = MagicMock()
        tensors.keys.return_value = ["1", "2", "xxx.text_model.position_embedding.weight", "3"]
        tensors.get_tensor.return_value = weight
        tokens = torch.tensor([[10, 20, 30]])
        result = _create_position_embedding(tensors, tokens)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertTrue(torch.equal(result, expected))

    def test_raises_if_no_position_embedding(self):
        tensors = MagicMock()
        tensors.keys.return_value = ["something_else.weight"]
        tokens = torch.tensor([[0, 1]])
        with self.assertRaises(ValueError):
            _create_position_embedding(tensors, tokens)