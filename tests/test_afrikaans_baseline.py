import unittest
import torch
from tools.train_afrikaans_baseline import pack_texts, make_model
from tools.compare_afrikaans_generators import diversity, score_text


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 0

    def __len__(self):
        return 16

    def __call__(self, texts, **kwargs):
        return {'input_ids': torch.tensor([[2, 3, 1, 0]] * len(texts)),
                'attention_mask': torch.tensor([[1, 1, 1, 0]] * len(texts))}

    def encode(self, text, **kwargs):
        return [2] * len(text)


class UniformModel:
    def __call__(self, **kwargs):
        class Result:
            loss = torch.tensor(2.0)
        return Result()


class BaselineTests(unittest.TestCase):
    def test_padding_and_bos(self):
        batch = pack_texts(Tokenizer(), ['text'], 4)
        self.assertEqual(batch['labels'].tolist(), [[-100, 2, 3, 1, -100]])
        self.assertEqual(batch['attention_mask'].tolist(), [[1, 1, 1, 1, 0]])

    def test_causal_model_loss_and_gradients(self):
        batch = pack_texts(Tokenizer(), ['text'], 4)
        model = make_model(Tokenizer(), 4, 64, 1)
        loss = model(**batch).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.transformer.wte.weight.grad).all())

    def test_sliding_windows_score_each_token_once(self):
        for length in (1, 7, 8, 9, 31):
            result = score_text(UniformModel(), Tokenizer(), 'x' * length, 'cpu', context=8)
            self.assertEqual(result['tokens'], length)
            self.assertAlmostEqual(result['nll'], 2 * length)

    def test_empty_is_explicit(self):
        self.assertIsNone(score_text(UniformModel(), Tokenizer(), '', 'cpu'))
        self.assertEqual(diversity(['', 'a a'])['empty_samples'], 1)
        self.assertEqual(diversity(['a a', 'a a'])['distinct_2'], 0.5)


if __name__ == '__main__':
    unittest.main()
