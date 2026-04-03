import os
import torch
from unittest import TestCase
from unittest.mock import patch, mock_open, call, MagicMock
from transformers import CLIPTokenizer
from pydiffuser.clip import (
    tokenize,
    embed,
    encode,
    _text_to_tokens,
    _break_up_tokens,
    _create_token_string_mapping,
    _create_token_embedding,
    _create_position_embedding,
    _get_clip_mask,
    _get_layer_tensors,
    _get_norm_tensors,
    _get_encoder_layer_numbers,
    _apply_mlp,
    _apply_normalization,
    _apply_attention,
    _apply_linear,
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


class EncodeTests(TestCase):

    @patch("torch.load")
    @patch("pydiffuser.clip._get_clip_mask")
    @patch("pydiffuser.clip.safetensors.safe_open")
    @patch("pydiffuser.clip._get_layer_tensors")
    @patch("pydiffuser.clip._get_norm_tensors")
    @patch("builtins.print")
    @patch("pydiffuser.clip._apply_attention")
    @patch("pydiffuser.clip._apply_mlp")
    @patch("pydiffuser.clip._apply_normalization")
    @patch("pydiffuser.clip.torch.save")
    def test_encode(self, mock_save, mock_norm, mock_mlp, mock_attention, mock_print, mock_norm_tensors, mock_layer_tensors, mock_safe_open, mock_clip_mask, mock_load):
        mock_load.return_value = torch.tensor([[1, 2, 3], [4, 5, 6]])
        mock_layer_tensors.return_value = {1: "TENSORS1", 2: "TENSORS2", 3: "TENSORS3"}
        mock_norm_tensors.return_value = {"weight": "NORM1_WEIGHT", "bias": "NORM1_BIAS"}
        mock_attention.side_effect = ("ATTN1", "ATTN2", "ATTN3")
        mock_mlp.side_effect = ("MLP1", "MLP2", "MLP3")
        encode("embedding.pt", "model.safetensors", "conditioning.pt")
        mock_load.assert_called_once_with("embedding.pt")
        self.assertTrue(torch.equal(mock_clip_mask.call_args[0][0], torch.tensor([[1., 2., 3.], [4., 5., 6.]])))
        mock_safe_open.assert_called_once_with("model.safetensors", framework="pt", device="cpu")
        mock_layer_tensors.assert_called_once_with(mock_safe_open.return_value.__enter__.return_value)
        self.assertEqual(mock_print.call_args_list, [call("Layer 1 of 3..."), call("Layer 2 of 3..."), call("Layer 3 of 3...")])
        self.assertTrue(torch.equal(mock_attention.call_args_list[0][0][0], torch.tensor([[1., 2., 3.], [4., 5., 6.]])))
        self.assertEqual(mock_attention.call_args_list[0][0][1], "TENSORS1")
        self.assertEqual(mock_attention.call_args_list[0][0][2], mock_clip_mask.return_value)
        self.assertEqual(mock_attention.call_args_list[1], call("MLP1", "TENSORS2", mock_clip_mask.return_value))
        self.assertEqual(mock_attention.call_args_list[2], call("MLP2", "TENSORS3", mock_clip_mask.return_value))
        self.assertEqual(mock_mlp.call_args_list[0], call("ATTN1", "TENSORS1"))
        self.assertEqual(mock_mlp.call_args_list[1], call("ATTN2", "TENSORS2"))
        self.assertEqual(mock_mlp.call_args_list[2], call("ATTN3", "TENSORS3"))
        mock_norm.assert_called_once_with("MLP3", weight="NORM1_WEIGHT", bias="NORM1_BIAS")
        mock_save.assert_called_once_with(mock_norm.return_value, "conditioning.pt")



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


class GetClipMaskTests(TestCase):

    def test_dim_2_tensor(self):
        embedding = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
        mask = _get_clip_mask(embedding)
        minus_inf = torch.finfo(torch.float32).min
        self.assertTrue(torch.equal(mask, torch.tensor([
            [0., minus_inf],
            [0., 0.]
        ])))

    def test_dim_3_tensor(self):
        embedding = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = _get_clip_mask(embedding)
        minus_inf = torch.finfo(torch.float32).min
        self.assertTrue(torch.equal(mask, torch.tensor([
            [0., minus_inf, minus_inf],
            [0., 0., minus_inf],
            [0., 0., 0.]
        ])))


class GetNormTensorsTests(TestCase):

    def test_get_norm_tensors(self):
        tensors = MagicMock()
        tensors.keys.return_value = [
            "cond_stage_model.transformer.text_model.encoder.final_layer_norm.weight",
            "cond_stage_model.transformer.text_model.encoder.final_layer_norm.bias",
            "cond_stage_model.layers.3.norm1.weight",
            "cond_stage_model.layers.3.norm1.bias",
            "sd_model.layers.10.norm1.bias",
        ]
        tensors.get_tensor.side_effect = lambda key: torch.tensor([len(key), len(key) * 2, len(key) * 3])
        norm_tensors = _get_norm_tensors(tensors)
        for k in norm_tensors:
            norm_tensors[k] = norm_tensors[k].tolist()
        self.assertEqual(norm_tensors, {
            "weight": [71, 142, 213],
            "bias": [69, 138, 207],
        })

    def test_all_norm_tensors_must_be_present(self):
        tensors = MagicMock()
        tensors.keys.return_value = [
            "cond_stage_model.transformer.text_model.encoder.final_layer_norm.weight",
            "cond_stage_model.layers.3.norm1.weight",
            "cond_stage_model.layers.3.norm1.bias",
            "sd_model.layers.10.norm1.bias",
        ]
        tensors.get_tensor.side_effect = lambda key: torch.tensor([len(key), len(key) * 2, len(key) * 3])
        with self.assertRaises(ValueError) as context:
            _get_norm_tensors(tensors)
        self.assertEqual(str(context.exception), "Final norm bias could not be found in the model")


class GetLayerTensorsTests(TestCase):

    @patch("pydiffuser.clip._get_encoder_layer_numbers")
    def test_get_layer_tensors(self, mock_get_layer_numbers):
        mock_get_layer_numbers.return_value = [3, 4, 5]
        tensors = MagicMock()
        tensors.keys.return_value = [
            "cond_stage_model.layers.3.norm1.weight",
            "cond_stage_model.layers.3.norm1.bias",
            "cond_stage_model.layers.3.norm2.weight",
            "cond_stage_model.layers.3.norm2.bias",
            "cond_stage_model.layers.3.mlp.fc1.weight",
            "cond_stage_model.layers.3.mlp.fc1.bias",
            "cond_stage_model.layers.3.mlp.fc2.weight",
            "cond_stage_model.layers.3.mlp.fc2.bias",
            "cond_stage_model.layers.3.attn.q_proj.weight",
            "cond_stage_model.layers.3.attn.q_proj.bias",
            "cond_stage_model.layers.3.attn.k_proj.weight",
            "cond_stage_model.layers.3.attn.k_proj.bias",
            "cond_stage_model.layers.3.attn.v_proj.weight",
            "cond_stage_model.layers.3.attn.v_proj.bias",
            "cond_stage_model.layers.3.attn.out_proj.weight",
            "cond_stage_model.layers.3.attn.out_proj.bias",
            "cond_stage_model.layers.4.norm1.weight",
            "cond_stage_model.layers.4.norm1.bias",
            "cond_stage_model.layers.4.norm2.weight",
            "cond_stage_model.layers.4.norm2.bias",
            "cond_stage_model.layers.4.mlp.fc1.weight",
            "cond_stage_model.layers.4.mlp.fc1.bias",
            "cond_stage_model.layers.4.mlp.fc2.weight",
            "cond_stage_model.layers.4.mlp.fc2.bias",
            "cond_stage_model.layers.4.attn.q_proj.weight",
            "cond_stage_model.layers.4.attn.q_proj.bias",
            "cond_stage_model.layers.4.attn.k_proj.weight",
            "cond_stage_model.layers.4.attn.k_proj.bias",
            "cond_stage_model.layers.4.attn.v_proj.weight",
            "cond_stage_model.layers.4.attn.v_proj.bias",
            "cond_stage_model.layers.4.attn.out_proj.weight",
            "cond_stage_model.layers.4.attn.out_proj.bias",
            "cond_stage_model.layers.5.norm1.weight",
            "cond_stage_model.layers.5.norm1.bias",
            "cond_stage_model.layers.5.norm2.weight",
            "cond_stage_model.layers.5.norm2.bias",
            "cond_stage_model.layers.5.mlp.fc1.weight",
            "cond_stage_model.layers.5.mlp.fc1.bias",
            "cond_stage_model.layers.5.mlp.fc2.weight",
            "cond_stage_model.layers.5.mlp.fc2.bias",
            "cond_stage_model.layers.5.attn.q_proj.weight",
            "cond_stage_model.layers.5.attn.q_proj.bias",
            "cond_stage_model.layers.5.attn.k_proj.weight",
            "cond_stage_model.layers.5.attn.k_proj.bias",
            "cond_stage_model.layers.5.attn.v_proj.weight",
            "cond_stage_model.layers.5.attn.v_proj.bias",
            "cond_stage_model.layers.5.attn.out_proj.weight",
            "cond_stage_model.layers.5.attn.out_proj.bias",
            "sd_model.layers.10.norm1.bias",
        ]
        tensors.get_tensor.side_effect = lambda key: torch.tensor([len(key), len(key) * 2, len(key) * 3])
        layer_tensors = _get_layer_tensors(tensors)
        for n in layer_tensors:
            for key in layer_tensors[n]:
                layer_tensors[n][key] = layer_tensors[n][key].tolist()
        self.assertEqual(layer_tensors, {
            3: {
                "norm1_weight": [38, 76, 114],
                "norm1_bias": [36, 72, 108],
                "norm2_weight": [38, 76, 114],
                "norm2_bias": [36, 72, 108],
                "mlp1_weight": [40, 80, 120],
                "mlp1_bias": [38, 76, 114],
                "mlp2_weight": [40, 80, 120],
                "mlp2_bias": [38, 76, 114],
                "attn_q_weight": [44, 88, 132],
                "attn_q_bias": [42, 84, 126],
                "attn_k_weight": [44, 88, 132],
                "attn_k_bias": [42, 84, 126],
                "attn_v_weight": [44, 88, 132],
                "attn_v_bias": [42, 84, 126],
                "attn_out_weight": [46, 92, 138],
                "attn_out_bias": [44, 88, 132]
            },
            4: {
                "norm1_weight": [38, 76, 114],
                "norm1_bias": [36, 72, 108],
                "norm2_weight": [38, 76, 114],
                "norm2_bias": [36, 72, 108],
                "mlp1_weight": [40, 80, 120],
                "mlp1_bias": [38, 76, 114],
                "mlp2_weight": [40, 80, 120],
                "mlp2_bias": [38, 76, 114],
                "attn_q_weight": [44, 88, 132],
                "attn_q_bias": [42, 84, 126],
                "attn_k_weight": [44, 88, 132],
                "attn_k_bias": [42, 84, 126],
                "attn_v_weight": [44, 88, 132],
                "attn_v_bias": [42, 84, 126],
                "attn_out_weight": [46, 92, 138],
                "attn_out_bias": [44, 88, 132]
            },
            5: {
                "norm1_weight": [38, 76, 114],
                "norm1_bias": [36, 72, 108],
                "norm2_weight": [38, 76, 114],
                "norm2_bias": [36, 72, 108],
                "mlp1_weight": [40, 80, 120],
                "mlp1_bias": [38, 76, 114],
                "mlp2_weight": [40, 80, 120],
                "mlp2_bias": [38, 76, 114],
                "attn_q_weight": [44, 88, 132],
                "attn_q_bias": [42, 84, 126],
                "attn_k_weight": [44, 88, 132],
                "attn_k_bias": [42, 84, 126],
                "attn_v_weight": [44, 88, 132],
                "attn_v_bias": [42, 84, 126],
                "attn_out_weight": [46, 92, 138],
                "attn_out_bias": [44, 88, 132]
            }
        })
    
    @patch("pydiffuser.clip._get_encoder_layer_numbers")
    def test_all_layers_must_be_present(self, mock_get_layer_numbers):
        mock_get_layer_numbers.return_value = [3, 4, 5]
        tensors = MagicMock()
        tensors.keys.return_value = [
            "cond_stage_model.layers.3.norm1.weight",
            "cond_stage_model.layers.3.norm1.bias",
            "cond_stage_model.layers.3.norm2.weight",
            "cond_stage_model.layers.3.norm2.bias",
            "cond_stage_model.layers.3.mlp.fc1.weight",
            "cond_stage_model.layers.3.mlp.fc1.bias",
            "cond_stage_model.layers.3.mlp.fc2.weight",
            "cond_stage_model.layers.3.attn.q_proj.weight",
            "cond_stage_model.layers.3.attn.q_proj.bias",
            "cond_stage_model.layers.3.attn.k_proj.weight",
            "cond_stage_model.layers.3.attn.k_proj.bias",
            "cond_stage_model.layers.3.attn.v_proj.weight",
            "cond_stage_model.layers.3.attn.v_proj.bias",
            "cond_stage_model.layers.3.attn.out_proj.weight",
            "cond_stage_model.layers.3.attn.out_proj.bias",
            "cond_stage_model.layers.4.norm1.weight",
            "cond_stage_model.layers.4.norm1.bias",
            "cond_stage_model.layers.4.norm2.weight",
            "cond_stage_model.layers.4.norm2.bias",
            "cond_stage_model.layers.4.mlp.fc1.weight",
            "cond_stage_model.layers.4.mlp.fc1.bias",
            "cond_stage_model.layers.4.mlp.fc2.weight",
            "cond_stage_model.layers.4.mlp.fc2.bias",
            "cond_stage_model.layers.4.attn.q_proj.weight",
            "cond_stage_model.layers.4.attn.q_proj.bias",
            "cond_stage_model.layers.4.attn.k_proj.weight",
            "cond_stage_model.layers.4.attn.k_proj.bias",
            "cond_stage_model.layers.4.attn.v_proj.weight",
            "cond_stage_model.layers.4.attn.v_proj.bias",
            "cond_stage_model.layers.4.attn.out_proj.weight",
            "cond_stage_model.layers.4.attn.out_proj.bias",
            "cond_stage_model.layers.5.norm1.weight",
            "cond_stage_model.layers.5.norm1.bias",
            "cond_stage_model.layers.5.norm2.weight",
            "cond_stage_model.layers.5.norm2.bias",
            "cond_stage_model.layers.5.mlp.fc1.weight",
            "cond_stage_model.layers.5.mlp.fc1.bias",
            "cond_stage_model.layers.5.mlp.fc2.weight",
            "cond_stage_model.layers.5.mlp.fc2.bias",
            "cond_stage_model.layers.5.attn.q_proj.weight",
            "cond_stage_model.layers.5.attn.q_proj.bias",
            "cond_stage_model.layers.5.attn.k_proj.weight",
            "cond_stage_model.layers.5.attn.k_proj.bias",
            "cond_stage_model.layers.5.attn.v_proj.weight",
            "cond_stage_model.layers.5.attn.v_proj.bias",
            "cond_stage_model.layers.5.attn.out_proj.weight",
            "cond_stage_model.layers.5.attn.out_proj.bias",
            "sd_model.layers.10.norm1.bias",
        ]
        tensors.get_tensor.side_effect = lambda key: torch.tensor([len(key), len(key) * 2, len(key) * 3])
        with self.assertRaises(ValueError) as context:
            _get_layer_tensors(tensors)
        self.assertEqual(str(context.exception), "Layer 3 mlp2_bias could not be found in the model")


class GetEncoderLayerNumbersTests(TestCase):

    def test_get_encoder_layer_numbers(self):
        tensors = MagicMock()
        tensors.keys.return_value = [
            "cond_stage_model.layers.1.norm1.weight",
            "cond_stage_model.layers.1.norm1.bias",
            "cond_stage_model.layers.4.norm1.bias",
            "cond_stage_Model.layers.3.norm1.bias",
            "cond_stage_model.layers.20.norm1.bias",
            "sd_model.layers.10.norm1.bias",
            "xxx"
        ]
        result = _get_encoder_layer_numbers(tensors)
        self.assertEqual(result, [1, 3, 4, 20])


class ApplyAttentionTests(TestCase):

    @patch("pydiffuser.clip._apply_normalization")
    @patch("pydiffuser.clip._apply_linear")
    def test_can_apply_attention(self, mock_linear, mock_norm):
        conditioning = torch.tensor([
            [[100., 200., 300., 100., 200., 300., 100., 200., 300., 100., 200., 300.]] * 3,
            [[400., 500., 600., 400., 500., 600., 400., 500., 600., 400., 500., 600.]] * 3,
        ])
        tensors = {
            "attn_q_weight": "ATTN_Q_WEIGHT",
            "attn_q_bias": "ATTN_Q_BIAS",
            "attn_k_weight": "ATTN_K_WEIGHT",
            "attn_k_bias": "ATTN_K_BIAS",
            "attn_v_weight": "ATTN_V_WEIGHT",
            "attn_v_bias": "ATTN_V_BIAS",
            "attn_out_weight": "ATTN_OUT_WEIGHT",
            "attn_out_bias": "ATTN_OUT_BIAS",
            "norm1_weight": "NORM1_WEIGHT",
            "norm1_bias": "NORM1_BIAS",
        }
        mock_norm.return_value = torch.tensor([
            [[5., 10., 15., 5., 10., 15., 5., 10., 15., 5., 10., 15.]] * 3,
            [[20., 25., 30., 20., 25., 30., 20., 25., 30., 20., 25., 30.]] * 3,
        ])
        mock_linear.side_effect = [
            torch.tensor([
                [[11., 21., 31., 11., 21., 31., 11., 21., 31., 11., 21., 31.]] * 3,
                [[41., 51., 61., 41., 51., 61., 41., 51., 61., 41., 51., 61.]] * 3,
            ]),
            torch.tensor([
                [[12., 22., 32., 12., 22., 32., 12., 22., 32., 12., 22., 32.]] * 3,
                [[42., 52., 62., 42., 52., 62., 42., 52., 62., 42., 52., 62.]] * 3,
            ]),
            torch.tensor([
                [[13., 23., 33., 13., 23., 33., 13., 23., 33., 13., 23., 33.]] * 3,
                [[43., 53., 63., 43., 53., 63., 43., 53., 63., 43., 53., 63.]] * 3,
            ]),
            torch.tensor([
                [[40., 50., 60., 40., 50., 60., 40., 50., 60., 40., 50., 60.]] * 3,
                [[70., 80., 90., 70., 80., 90., 70., 80., 90., 70., 80., 90.]] * 3,
            ]),
        ]
        mask = torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        result = _apply_attention(conditioning, tensors, mask)
        self.assertEqual(result.size(), (2, 3, 12))
        self.assertEqual(
            [round(float(v), 4) for v in result.reshape(-1).tolist()],
            [140, 250, 360, 140, 250, 360, 140, 250, 360, 140, 250, 360, 140, 250, 360, 140, 250, 360, 140, 250, 360, 140, 250, 360, 140, 250, 360, 140, 250, 360, 140, 250, 360, 140, 250, 360, 470, 580, 690, 470, 580, 690, 470, 580, 690, 470, 580, 690, 470, 580, 690, 470, 580, 690, 470, 580, 690, 470, 580, 690, 470, 580, 690, 470, 580, 690, 470, 580, 690, 470, 580, 690]
        )
        self.assertTrue(torch.equal(mock_norm.call_args_list[0][0][0], conditioning))
        self.assertEqual(mock_norm.call_args_list[0][0][1], "NORM1_WEIGHT")
        self.assertEqual(mock_norm.call_args_list[0][0][2], "NORM1_BIAS")
        self.assertTrue(torch.equal(mock_linear.call_args_list[0][0][0], mock_norm.return_value))
        self.assertEqual(mock_linear.call_args_list[0][0][1], "ATTN_Q_WEIGHT")
        self.assertEqual(mock_linear.call_args_list[0][0][2], "ATTN_Q_BIAS")
        self.assertTrue(torch.equal(mock_linear.call_args_list[1][0][0], mock_norm.return_value))
        self.assertEqual(mock_linear.call_args_list[1][0][1], "ATTN_K_WEIGHT")
        self.assertEqual(mock_linear.call_args_list[1][0][2], "ATTN_K_BIAS")
        self.assertTrue(torch.equal(mock_linear.call_args_list[2][0][0], mock_norm.return_value))
        self.assertEqual(mock_linear.call_args_list[2][0][1], "ATTN_V_WEIGHT")
        self.assertEqual(mock_linear.call_args_list[2][0][2], "ATTN_V_BIAS")
        self.assertEqual(mock_linear.call_args_list[3][0][0].size(), (2, 3, 12))
        self.assertEqual(
            [round(float(v), 4) for v in mock_linear.call_args_list[3][0][0].reshape(-1).tolist()],
            [13.0, 23.0, 33.0, 13.0, 23.0, 33.0, 13.0, 23.0, 33.0, 13.0, 23.0, 33.0, 13.0, 23.0, 33.0, 13.0, 23.0, 33.0, 13.0, 23.0, 33.0, 13.0, 23.0, 33.0, 13.0, 23.0, 33.0, 13.0, 23.0, 33.0, 13.0, 23.0, 33.0, 13.0, 23.0, 33.0, 43.0, 53.0, 63.0, 43.0, 53.0, 63.0, 43.0, 53.0, 63.0, 43.0, 53.0, 63.0, 43.0, 53.0, 63.0, 43.0, 53.0, 63.0, 43.0, 53.0, 63.0, 43.0, 53.0, 63.0, 43.0, 53.0, 63.0, 43.0, 53.0, 63.0, 43.0, 53.0, 63.0, 43.0, 53.0, 63.0]
        )


class ApplyMlpTests(TestCase):

    @patch("pydiffuser.clip._apply_normalization")
    @patch("pydiffuser.clip._apply_linear")
    def test_can_apply_mlp(self, mock_linear, mock_norm):
        conditioning = torch.tensor([[100., 200., 300.], [400., 500., 600.]])
        tensors = {
            "norm2_weight": "NORM2_WEIGHT",
            "norm2_bias": "NORM2_BIAS",
            "mlp1_weight": "MLP1_WEIGHT",
            "mlp1_bias": "MLP1_BIAS",
            "mlp2_weight": "MLP2_WEIGHT",
            "mlp2_bias": "MLP2_BIAS",
        }
        mock_norm.return_value = torch.tensor([[5., 10., 15.], [20., 25., 30.]])
        mock_linear.side_effect = [
            torch.tensor([[1., -2., 3.], [4., 5., 6.]]),
            torch.tensor([[70., 80., 90.], [100., 110., 120.]]),
        ]
        result = _apply_mlp(conditioning, tensors)
        self.assertTrue(torch.equal(result, torch.tensor([[170., 280., 390.], [500, 610, 720]])))
        mock_norm.assert_called_once_with(conditioning, "NORM2_WEIGHT", "NORM2_BIAS")
        self.assertTrue(torch.equal(mock_linear.call_args_list[0][0][0], torch.tensor([[5., 10., 15.], [20., 25., 30.]])))
        self.assertEqual(mock_linear.call_args_list[0][0][1], "MLP1_WEIGHT")
        self.assertEqual(mock_linear.call_args_list[0][0][2], "MLP1_BIAS")
        self.assertEqual(mock_linear.call_args_list[1][0][0].size(), (2, 3))
        self.assertEqual(
            [round(float(v), 4) for v in mock_linear.call_args_list[1][0][0].reshape(-1).tolist()],
            [0.8458, -0.0643, 2.9819, 3.9956, 4.9990, 5.9998]
        )
        self.assertEqual(mock_linear.call_args_list[1][0][1], "MLP2_WEIGHT")
        self.assertEqual(mock_linear.call_args_list[1][0][2], "MLP2_BIAS")


class ApplyNormalizationTests(TestCase):

    def test_can_apply_normalization(self):
        tensor = torch.tensor([[100., 200., 300.], [405., 500., 550.]])
        weights = torch.tensor([4., 5., 6.])
        bias = torch.tensor([16., 17., 18.])
        output = _apply_normalization(tensor, weights, bias)
        output = torch.round(output * 1000) / 1000
        self.assertEqual(output.shape, (2, 3))
        self.assertEqual(
            [round(float(v), 3) for v in output.reshape(-1).tolist()],
            [11.101, 17.0, 25.348, 10.679, 18.247, 24.485]
        )

class ApplyLinearTests(TestCase):

    def test_can_apply_linear(self):
        tensor = torch.tensor([[100., 200., 300.], [400., 500., 600.]])
        weights = torch.tensor([[4., 5., 6.], [7., 8., 9.], [10., 11., 12.], [13., 14., 15.]])
        bias = torch.tensor([16., 17., 18., 19.])
        output = _apply_linear(tensor, weights, bias)
        self.assertEqual(output.tolist(), [[3216., 5017., 6818., 8619.], [7716., 12217., 16718., 21219.]])