import torch


def get_tgt_mask(size) -> torch.tensor:
    mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

    return mask

def create_pad_mask(matrix: torch.tensor, pad_token: int) -> torch.tensor:
    # Create (batch_size, seq_len) tensor
    seq = matrix[:, :, 0]
    return (seq == pad_token)