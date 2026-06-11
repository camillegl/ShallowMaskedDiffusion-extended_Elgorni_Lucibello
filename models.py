
import torch
from torch import nn

# defines: LinearBackbone, RandomFeatureScore, TensorBackbone


def act_str_to_fn(act: str):
    if act == "relu":
        return nn.ReLU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "identity":
        return nn.Identity()
    elif act == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation function {act}. Supported: relu, tanh, sigmoid, identity, gelu.")


class MaskedDiffusionBackbone(nn.Module):
    """
    Base class for masked diffusion backbones. Defines the interface.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")

    def freeze_mask_weights(self):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def train_log(self, logger, batch_idx):
        pass


class LinearBackbone(MaskedDiffusionBackbone):
    """Linear backbone with two learnable LxL matrices W and V. No biases.

    Forward(x): x is (batch, L). Compute x @ W.T + (x==0).float() @ V.T
    """

    def __init__(self, L: int, mask_index: int = 0, bias: bool = False):
        super().__init__()
        self.L = L
        self.mask_index = mask_index
        # Use weight matrices without biases to match Julia implementation
        self.W = nn.Parameter(torch.randn(L, L) / L**0.5) # weights for the unmasked input, assuming it is not categorical
        self.V = nn.Parameter(torch.zeros(L, L)) # weights for the mask
        if bias:
            self.b = nn.Parameter(torch.zeros(L))
        else:
            self.b = None

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        # xt: (batch_size, L). masked positions are set to 0 in the input.
        mask = (xt == self.mask_index).to(xt.dtype)  # (batch, L)
        out = xt @ self.W.t() + mask @ self.V.t()
        if self.b is not None:
            out = out + self.b
        return out

    @torch.no_grad()
    def set_hebbian_weights(self, train_data: torch.Tensor, scale: float = None, zero_diagonal: bool = False):
        """Replace W with the Hebbian matrix built from the training data.

        Parameters
        ----------
        train_data:
            Tensor of shape [num_samples, L] containing the training patterns.
        scale:
            Optional multiplicative scale applied to the Hebbian matrix.
            If None, the raw sum over samples is used.
        zero_diagonal:
            If True, zero out the diagonal after constructing W.
        """
        if train_data.ndim != 2:
            raise ValueError(f"train_data must have shape [num_samples, L], got {tuple(train_data.shape)}")
        if train_data.shape[1] != self.L:
            raise ValueError(f"train_data has length {train_data.shape[1]}, but model was built with L={self.L}")

        hebbian = train_data.t() @ train_data
        if scale is not None:
            hebbian = hebbian / self.L * scale
        if zero_diagonal:
            hebbian = hebbian.clone()
            hebbian.fill_diagonal_(0.0)

        self.W.copy_(hebbian.to(device=self.W.device, dtype=self.W.dtype))
        self.W.requires_grad = False

    @torch.no_grad()
    def train_log(self, logger, batch_idx):
        W2mean = (self.W**2).mean().item() * self.L
        V2mean = (self.V**2).mean().item() * self.L
        logger("train/qW", W2mean, on_step=True, on_epoch=False, prog_bar=False)
        logger("train/qV", V2mean, on_step=True, on_epoch=False, prog_bar=False)
        
    def freeze_mask_weights(self):
        self.V.requires_grad = False


class RandomFeatureScore(MaskedDiffusionBackbone):
    def __init__(self, L: int, mask_index: int = 0, n_hidden: int = 100, act="relu", bias: bool = False):
        super().__init__()
        self.L = L
        self.mask_index = mask_index
        self.n_hidden = n_hidden
        self.act = act_str_to_fn(act)
        self.W1 = nn.Parameter(torch.randn(n_hidden, L) / L**0.5, requires_grad=False) # fixed random weights
        self.V1 = nn.Parameter(torch.zeros(n_hidden, L), requires_grad=False) # fixed random weights
        self.W2 = nn.Parameter(torch.randn(L, n_hidden) / n_hidden**0.5) # learnable weights
        if bias:
            self.b = nn.Parameter(torch.zeros(L))
        else:
            self.b = None
            
    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        out = xt @ self.W1.t() + (xt == self.mask_index).float() @ self.V1.t()
        out = self.act(out) @ self.W2.t()
        if self.b is not None:
            out = out + self.b
        return out
    
    def freeze_mask_weights(self):
        self.V1.requires_grad = False

class TensorBackbone(MaskedDiffusionBackbone):
    """Like LinearBackbone but with different weight matrices for different levels of masking.
    The number of levels is nbins, and the bin for a sample is determined by the number of unmasked tokens.
    
    # Note: for performance reason, the parametrization is such that we don't need to use transposes in the forward pass,
    as it is usually done in linear layers.
    """

    def __init__(self, L: int, nbins: int = 8, mask_index: int = 0, vectorized = None, bias: bool = False):
        super().__init__()
        self.L = L
        self.mask_index = mask_index
        self.vectorized = vectorized
        self.nbins = nbins
        self.bins = self._bin_counts(torch.arange(L))  # (L,)
        # Use weight matrices without biases to match Julia implementation
        self.W = nn.Parameter(torch.randn(self.nbins, L, L) / L**0.5) # weights for the unmasked input, assuming it is not categorical
        self.V = nn.Parameter(torch.zeros(self.nbins, L, L)) # weights for the mask
        if bias:
            self.b = nn.Parameter(torch.zeros(L))
        else:
            self.b = None
    
    def _bin_counts(self, n_unmasked: torch.Tensor) -> torch.Tensor:
        b = (n_unmasked * self.nbins) // self.L
        b = torch.clamp(b, 0, self.nbins - 1)
        return b.to(torch.long)

    def forward(self, xt: torch.Tensor, vectorized = None) -> torch.Tensor:
        B, L = xt.shape
        assert L == self.L, f"xt has length {L}, but model was built with L={self.L}"
        
        # Build mask and counts
        mask = (xt == self.mask_index).to(xt.dtype)             # (B, L)
        n_unmasked = (self.L - mask.sum(dim=1)).to(torch.long)  # (B,)
        bin_idx = self._bin_counts(n_unmasked)                  # (B,)
        uniq, inv = torch.unique(bin_idx, return_inverse=True)
        if vectorized is None:
            if self.vectorized is None:
                vectorized = len(uniq) <= 4 # heuristic: use vectorized if <=4 bins are present in the batch
            else:
                vectorized = self.vectorized

        assert vectorized in [True, False]

        # Compute output. We have three possible strategies:
        # 1) if all samples fall in the same bin, we can do a single matrix multiplication
        # 2) if vectorized=False, we can do the computation sequentially for each bin (memory-friendly but slow)
        # 3) if vectorized=True, we can do the computation with batch matrix multiplication (fast but memory-hungry)
        
        if len(uniq) == 1:
            # Fast path: if all samples fall in the same bin
            k = bin_idx[0]
            Wk = self.W[k]  # (L, L)
            Vk = self.V[k]  # (L, L)
            return xt @ Wk + mask @ Vk  # (B, L)
        elif not vectorized:
            # Do the computation sequentially for each bin (memory-friendly but slow)
            pieces, idxs = [], []
            for i, k in enumerate(uniq):
                sel_idx = torch.nonzero(inv == i, as_tuple=False).squeeze(1)  # (Bk,)
                xk, mk = xt[sel_idx], mask[sel_idx]                           # (Bk, L)
                Wk, Vk = self.W[k], self.V[k]                                 # (L, L)
                yk = xk @ Wk + mk @ Vk                               # (Bk, L)
                pieces.append(yk)
                idxs.append(sel_idx)

            y_all = torch.cat(pieces, dim=0)          # (B, L) but permuted
            idx_all = torch.cat(idxs, dim=0)          # (B,)
            out = xt.new_empty(B, L)
            out = out.index_copy(0, idx_all, y_all)   # (B, L)
        else:
            # vectorized path with batch matrix multiplication (fast but memory-hungry)
            W_sel = self.W.index_select(0, bin_idx) # (B, L, L)
            V_sel = self.V.index_select(0, bin_idx)
            out = torch.bmm(xt.unsqueeze(1), W_sel).squeeze(1) # (B, L)
            out += torch.bmm(mask.unsqueeze(1), V_sel).squeeze(1)
        if self.b is not None:
            out = out + self.b
        return out
    
    def freeze_mask_weights(self):
        self.V.requires_grad = False