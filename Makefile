PYTHON = python3

.PHONY: test test-mytorch test-masks test-positional test-sublayers test-decoder test-transformer clean

test: test-mytorch test-masks test-positional test-sublayers test-decoder test-transformer
	@echo "All tests completed"

test-mytorch:
	@echo "Testing MyTorch Components"
	@$(PYTHON) -c "import sys; sys.path.append('.'); from tests.test_mytorch_linear import test_linear; test_linear()"
	@$(PYTHON) -c "import sys; sys.path.append('.'); from tests.test_mytorch_softmax import test_softmax; test_softmax()"
	@$(PYTHON) -c "import sys; sys.path.append('.'); from tests.test_mytorch_scaled_dot_product_attention import test_scaled_dot_product_attention; test_scaled_dot_product_attention()"
	@$(PYTHON) -c "import sys; sys.path.append('.'); from tests.test_mytorch_multi_head_attention import test_multi_head_attention; test_multi_head_attention()"

test-masks:
	@echo "Testing Mask Functions"
	@$(PYTHON) -c "import sys, torch; sys.path.append('.'); import importlib.util; spec = importlib.util.spec_from_file_location('masks', 'hw4lib/model/masks.py'); masks = importlib.util.module_from_spec(spec); spec.loader.exec_module(masks); from tests.test_mask_causal import test_mask_causal; from tests.test_mask_padding import test_mask_padding; test_mask_causal(masks.CausalMask); test_mask_padding(masks.PadMask)"

test-positional:
	@echo "Testing PositionalEncoding"
	@$(PYTHON) -c "import sys, torch; sys.path.append('.'); import importlib.util; spec = importlib.util.spec_from_file_location('pe', 'hw4lib/model/positional_encoding.py'); pe_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(pe_module); from tests.test_positional_encoding import test_positional_encoding; test_positional_encoding(pe_module.PositionalEncoding)"

test-sublayers:
	@echo "Testing Sublayers"
	@$(PYTHON) -c "import sys; sys.path.append('.'); from tests.test_sublayer_selfattention import test_selfattention_sublayer; test_selfattention_sublayer()"
	@$(PYTHON) -c "import sys; sys.path.append('.'); from tests.test_sublayer_feedforward import test_feedforward_sublayer; test_feedforward_sublayer()"

test-decoder:
	@echo "Testing Decoder Layers"
	@$(PYTHON) -c "import sys; sys.path.append('.'); from tests.test_decoderlayer_selfattention import test_decoderlayer_selfattention; test_decoderlayer_selfattention()"

test-transformer:
	@echo "Testing Transformer"
	@$(PYTHON) -c "import sys; sys.path.append('.'); from tests.test_transformer_decoder_only import test_transformer_decoder_only; test_transformer_decoder_only()"

clean:
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +