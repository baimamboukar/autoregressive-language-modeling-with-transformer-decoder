.PHONY: test test-hw4p1 test-mytorch test-model test-all test-hw4p2

test:
	@PYTHONPATH=. python3 -W ignore tests/test_mytorch_linear.py
	@PYTHONPATH=. python3 -W ignore tests/test_mytorch_softmax.py
	@PYTHONPATH=. python3 -W ignore tests/test_mytorch_scaled_dot_product_attention.py
	@PYTHONPATH=. python3 -W ignore tests/test_mytorch_multi_head_attention.py
	@PYTHONPATH=. python3 -W ignore tests/test_dataset_lm.py
	@PYTHONPATH=. python3 -W ignore tests/test_mask_causal.py
	@PYTHONPATH=. python3 -W ignore tests/test_mask_padding.py
	@PYTHONPATH=. python3 -W ignore tests/test_positional_encoding.py
	@PYTHONPATH=. python3 -W ignore tests/test_sublayer_selfattention.py
	@PYTHONPATH=. python3 -W ignore tests/test_sublayer_feedforward.py
	@PYTHONPATH=. python3 -W ignore tests/test_decoderlayer_selfattention.py
	@PYTHONPATH=. python3 -W ignore tests/test_transformer_decoder_only.py
	@PYTHONPATH=. python3 -W ignore tests/test_decoding.py
