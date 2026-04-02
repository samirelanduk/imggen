import os
from unittest import TestCase
from unittest.mock import patch, mock_open, call
from transformers import CLIPTokenizer
from pydiffuser.clip import (
    tokenize,
    _text_to_tokens,
    _break_up_tokens,
    _create_token_string_mapping,
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