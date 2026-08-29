import unittest

from cot_faithfulness.model_loader import verify_decoder_layers


class FakeBase:
    layers = [object(), object(), object()]


class FakeModel:
    model = FakeBase()


class ModelLoaderTests(unittest.TestCase):
    def test_verifies_decoder_layer_path(self):
        verify_decoder_layers(FakeModel(), (0, 2))

    def test_rejects_out_of_range_layer(self):
        with self.assertRaises(RuntimeError):
            verify_decoder_layers(FakeModel(), (3,))


if __name__ == "__main__":
    unittest.main()
