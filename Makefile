PYTHON = python3

.PHONY: test test-basic test-masks test-positional clean

test: test-basic test-masks
	@echo "All compatible tests completed"

test-basic:
	@echo "Testing Basic Components"
	@$(PYTHON) -c "import sys; sys.path.append('.'); from tests.test_mytorch_linear import test_linear; test_linear()"
	@$(PYTHON) -c "import sys; sys.path.append('.'); from tests.test_mytorch_softmax import test_softmax; test_softmax()"

test-masks:
	@echo "Testing Mask Functions"
	@$(PYTHON) -c "import sys, torch; sys.path.append('.'); import importlib.util; spec = importlib.util.spec_from_file_location('masks', 'hw4lib/model/masks.py'); masks = importlib.util.module_from_spec(spec); spec.loader.exec_module(masks); from tests.test_mask_causal import test_mask_causal; from tests.test_mask_padding import test_mask_padding; test_mask_causal(masks.CausalMask); test_mask_padding(masks.PadMask)"

test-positional:
	@echo "Testing PositionalEncoding"
	@$(PYTHON) -c "import sys, torch; sys.path.append('.'); import importlib.util; spec = importlib.util.spec_from_file_location('pe', 'hw4lib/model/positional_encoding.py'); pe_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(pe_module); from tests.test_positional_encoding import test_positional_encoding; test_positional_encoding(pe_module.PositionalEncoding)"

clean:
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +