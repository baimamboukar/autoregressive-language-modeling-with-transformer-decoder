import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # Initialize sequences and scores
        batch_size, seq_len = x.size()
        sequences = x.clone()  # Start with input sequence
        scores = torch.zeros(batch_size, device=self.device)  # Track cumulative scores

        # Generate tokens until max_length or EOS
        for _ in range(self.max_length - seq_len):
            # Get logits for next token
            logits = self.score_fn(sequences)  # (batch_size, vocab_size)

            # Apply repeat penalty
            logits = self._apply_repeat_penalty(logits, sequences, repeat_penalty)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Get probabilities and select most likely token (greedy)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            # Add to sequences
            sequences = torch.cat([sequences, next_token], dim=1)

            # Update scores with log probability
            token_scores = torch.log(probs.gather(1, next_token).squeeze(1))
            scores += token_scores

            # Check for EOS tokens
            if torch.all(next_token.squeeze(1) == self.tokenizer.eos_id):
                break

        return sequences, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length) where each sequence in a beam set is sorted by score
             - scores is of shape (batch_size, beam_width)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # Initialize beam search
        batch_size, seq_len = x.size()
        vocab_size = self.tokenizer.vocab_size

        # For beam search, we need to handle the initial expansion differently
        # Start with input sequences and expand step by step
        current_seqs = x  # (batch_size, seq_len)
        current_beam_scores = torch.zeros(batch_size, device=self.device)  # Initialize here

        # Generate tokens until max_length
        for step in range(self.max_length - seq_len):
            # Collect candidates for this step
            all_candidates = []
            all_scores = []

            # If this is the first step, start from single input sequence per batch
            if step == 0:
                sequences_to_expand = current_seqs  # (batch_size, seq_len)
                scores_to_use = current_beam_scores  # (batch_size,)
            else:
                # Subsequent steps: expand all current beams
                sequences_to_expand = current_seqs.view(-1, current_seqs.size(-1))  # (batch_size * beam_width, seq_len)
                scores_to_use = current_beam_scores.view(-1)  # (batch_size * beam_width,)

            for seq_idx in range(sequences_to_expand.size(0)):
                seq = sequences_to_expand[seq_idx:seq_idx+1]  # (1, seq_len)
                base_score = scores_to_use[seq_idx] if step > 0 else 0.0

                # Get next token probabilities
                logits = self.score_fn(seq).squeeze(0)  # (vocab_size,)

                # Apply repeat penalty and temperature
                if repeat_penalty != 1.0:
                    logits = self._apply_repeat_penalty(logits.unsqueeze(0), seq, repeat_penalty).squeeze(0)

                if temperature != 1.0:
                    logits = logits / temperature

                log_probs = torch.log_softmax(logits, dim=-1)

                # Get top-k candidates for this sequence
                top_k = beam_width if step == 0 else beam_width  # Allow beam_width candidates
                top_logprobs, top_tokens = torch.topk(log_probs, top_k)

                # Create candidates
                for k in range(top_k):
                    new_seq = torch.cat([seq.squeeze(0), top_tokens[k:k+1]], dim=0)
                    new_score = base_score + top_logprobs[k].item()
                    all_candidates.append(new_seq)
                    all_scores.append(new_score)

            # Process candidates for each batch separately
            if step == 0:
                # First step: select top beam_width from single sequence expansion per batch
                new_seq_len = x.size(-1) + 1
                current_seqs = torch.zeros(batch_size, beam_width, new_seq_len, device=self.device, dtype=torch.long)
                current_beam_scores = torch.zeros(batch_size, beam_width, device=self.device)

                # For step 0, we have one candidate set per batch item
                candidates_per_batch = beam_width  # We generated beam_width candidates from each input sequence

                for batch_idx in range(batch_size):
                    start_idx = batch_idx * candidates_per_batch
                    end_idx = start_idx + candidates_per_batch
                    batch_candidates = all_candidates[start_idx:end_idx]
                    batch_scores = all_scores[start_idx:end_idx]

                    # Select top beam_width for this batch
                    sorted_pairs = sorted(zip(batch_scores, batch_candidates), key=lambda x: x[0], reverse=True)
                    selected_pairs = sorted_pairs[:beam_width]

                    for beam_idx in range(beam_width):
                        if beam_idx < len(selected_pairs):
                            score, seq = selected_pairs[beam_idx]
                            current_seqs[batch_idx, beam_idx] = seq
                            current_beam_scores[batch_idx, beam_idx] = score
            else:
                # Subsequent steps: handle beam expansion
                candidates_per_batch = beam_width * beam_width  # Each beam generates beam_width candidates
                new_seq_len = current_seqs.size(-1) + 1
                new_seqs = torch.zeros(batch_size, beam_width, new_seq_len, device=self.device, dtype=torch.long)
                new_scores = torch.zeros(batch_size, beam_width, device=self.device)

                for batch_idx in range(batch_size):
                    start_idx = batch_idx * candidates_per_batch
                    end_idx = start_idx + candidates_per_batch
                    batch_candidates = all_candidates[start_idx:end_idx]
                    batch_scores = all_scores[start_idx:end_idx]

                    # Select top beam_width for this batch
                    sorted_pairs = sorted(zip(batch_scores, batch_candidates), key=lambda x: x[0], reverse=True)
                    selected_pairs = sorted_pairs[:beam_width]

                    for beam_idx in range(beam_width):
                        if beam_idx < len(selected_pairs):
                            score, seq = selected_pairs[beam_idx]
                            new_seqs[batch_idx, beam_idx] = seq
                            new_scores[batch_idx, beam_idx] = score

                current_seqs = new_seqs
                current_beam_scores = new_scores

            # Check for early termination
            if current_seqs.size(-1) >= self.max_length:
                break

            # Check if all sequences end with EOS
            last_tokens = current_seqs[:, :, -1]
            if torch.all(last_tokens == self.tokenizer.eos_id):
                break

        # Return final beams and scores
        beams = current_seqs
        beam_scores = current_beam_scores

        return beams, beam_scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check if any sequence has reached EOS 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]