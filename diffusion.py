import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from tqdm import tqdm

from models import LinearBackbone, RandomFeatureScore, TensorBackbone
        
class MaskedDiffusion(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.mask_index = 0
        bias = getattr(config, "bias", False)
        if config.model == "linear":
            self.backbone = LinearBackbone(config.L, mask_index=self.mask_index, bias=bias)
        elif config.model.startswith("rfs"):
            # e.g. model = "rfs10_tanh" means random features with 10x expansion and tanh activation
            expansion_factor_str, act_str = config.model[3:].split("_")
            expansion_factor = float(expansion_factor_str)
            n_hidden = int(round(config.L * expansion_factor))
            self.backbone = RandomFeatureScore(config.L, mask_index=self.mask_index, n_hidden=n_hidden, act=act_str, bias=bias)
        elif config.model.startswith("tensor"):
            nbins = int(config.model[6:])
            self.backbone = TensorBackbone(config.L, nbins=nbins, mask_index=self.mask_index, bias=bias)
        else:
            raise ValueError(f"Unknown model {config.model}. Supported: linear, rfs<expansion>_<activation>.")

        if getattr(config, "freeze_mask_weights", False):
            self.backbone.freeze_mask_weights()
        self.L = config.L
        self.t_val_list = [0.1, 0.5, 0.9, 1.0]

        self.l2coeff = 0.5 * config.l2reg / (self.L * config.alpha) # So that lambda from the replica calculation
                                                                       # matches l2reg
        # With respect to the replica calculation, both the data loss and the parameters have different normalizations.

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.backbone(xt)

    def training_step(self, batch, batch_idx):
        x0 = batch
        loss, acc = self._compute_loss(x0)
        l2loss = self.l2coeff * self.sqnorm() # normalize by L to keep scale consistent across different L
                                    # Also the loss is already normalized by L in _compute_loss
        tot_loss = loss + l2loss
        self.log("train/acc", acc, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/l2loss", l2loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/total_loss", tot_loss, on_step=True, on_epoch=False, prog_bar=True) 
        self.backbone.train_log(self.log, batch_idx) # log backbone-specific metrics
        return tot_loss

    def sqnorm(self):
        ms = 0.0
        for p in self.parameters():
            ms += (p**2).sum()
        return ms


    def _compute_loss(self, x0, t = None, mc_samples=1):
        # TODO: I've read somewhere that instead of weigthing with 1/t
        # using L / |mask| is less noisy, maybe I should change to that
        batch_size = x0.shape[0]

        loss = 0.0
        acc = 0.0
        for _ in range(mc_samples):
            if t is None:
                t = torch.rand(batch_size, device=self.device, dtype=torch.float32)
            if isinstance(t, float):
                t = torch.full((batch_size,), t, device=self.device, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(1).repeat(1, self.L) # [batch_size, L]
            assert t.shape == (batch_size, self.L)
            
            mask = torch.rand_like(x0) < t
            xt = x0.clone()
            xt[mask] = self.mask_index # xt has zeros where mask True, original x0 elsewhere
            
            logits = self(xt)       # [batch, L]
            logits = logits[mask]   # [num_masked]
            y = (x0[mask] + 1) / 2 # target y in {0,1}
            
            losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            weight = 1 / t[mask]
            loss += (losses * weight).sum() / (self.L * batch_size)

            yhat = logits > 0
            acc += (yhat == y).to(torch.float32).mean().item()
        
        return loss / mc_samples, acc / mc_samples

    def validation_step(self, batch, batch_idx):
        x0 = batch
        loss, acc = self._compute_loss(x0)            
        self.log("val/acc", acc, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x0 = batch
        batch_size, L = x0.shape
        mc_samples = 10
        
        ## Total loss and accuracy
        loss, acc = self._compute_loss(x0, mc_samples=mc_samples)

        self.log(f"test/acc", acc)
        self.log(f"test/loss", loss)
        
        ## Time-sliced loss and accuracy
        for t_val in self.t_val_list:
            loss, acc = self._compute_loss(x0, t=t_val, mc_samples=mc_samples)
            self.log(f"test/loss_t{t_val}", loss)
            self.log(f"test/acc_t{t_val}", acc)
            
        ## U-Turn Experiment
        for t_val in self.t_val_list:
            mask = torch.rand_like(x0) < t_val
            xt = x0.clone()
            xt[mask] = self.mask_index
            xnew = self.sample(batch_size, k=1, xt=xt)
            overlap = (xnew * x0).mean().item()
            self.log(f"test/uturn_overlap_t{t_val}", overlap)



    def configure_optimizers(self):
        ## L2 penalty is done in the loss, no weight decay here
        optimizer = torch.optim.AdamW(self.parameters(), 
                                      lr=self.config.lr, 
                                      weight_decay=0.0)
        

        ## scheduler: drop LR by factor 0.1 for the last 20% of training
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=[int(0.5 * self.config.epochs), int(0.8 * self.config.epochs)], 
                                                         gamma=0.5)
        return [optimizer], [scheduler]
        
        ## ALTERNATIVE
        # optimizer = torch.optim.ASGD(self.parameters(), lr=self.config.lr, weight_decay=0.0)  # lr=0.1 is a good default
        # return optimizer
    
    @torch.no_grad()
    def sample(self, nsamples: int = None, k: int = 1, xt = None) -> torch.Tensor:
        """Sample k tokens at a time until all tokens are unmasked.
        """
        if nsamples is None:
            assert xt is not None, "Either nsamples or xt must be provided."
            batch_size = xt.shape[0]
        else:
            batch_size = nsamples
        
        if xt is None:
            xt = torch.full((batch_size, self.L), self.mask_index, device=self.device).to(torch.float32)
        else:
            if xt.ndim == 2:
                assert xt.shape[0] == batch_size
                xt = xt.clone().to(self.device)
            elif xt.ndim == 1:
                assert xt.shape[0] == self.L
                xt = xt.unsqueeze(0).repeat(batch_size, 1).to(self.device)
            else:
                raise ValueError("xt must be of shape (L,) or (batch_size, L)")

        # Progress bar setup
        total_tokens = xt.numel()                 # batch_size * L
        # initialize progress bar with current number of unmasked tokens
        prev_unmasked = int((xt != self.mask_index).sum().item())
        
        if getattr(self.config, "pbar", True):
            pbar = tqdm(total=total_tokens-prev_unmasked, unit="tok", position=1, leave=False, desc="  Sampling")
            pbar.update(0)

        keep_sampling = prev_unmasked < total_tokens
        while keep_sampling:
            xt = self._sample_k_update(xt, k)
            curr_unmasked = int((xt != self.mask_index).sum().item())
            delta = curr_unmasked - prev_unmasked
            if getattr(self.config, "pbar", True):
                pbar.update(delta)
            prev_unmasked = curr_unmasked

            # stop when no masked tokens remain (i.e. everything is unmasked)
            keep_sampling = curr_unmasked < total_tokens

        if getattr(self.config, "pbar", True):
            pbar.close()
        return xt

    
    def _sample_k_update(self, xt, k):
        batch_size, L = xt.shape

        p_x0 = torch.sigmoid(self.forward(xt)) # [batch_size, L]

        mask = (xt == self.mask_index)          # [batch_size, L]

        if k == 1:
            # Vectorized fast path: one uniformly-chosen masked position per row.
            rows = mask.any(dim=1)
            if rows.any():
                pos = torch.multinomial(mask[rows].to(p_x0.dtype), 1).squeeze(1)  # [n_rows]
                q_s = p_x0[rows, pos]
                xt[rows.nonzero(as_tuple=True)[0], pos] = 2*torch.bernoulli(q_s) - 1
            return xt

        pos_to_unmask = []                     # list of posidx of length batch_size
        for b in range(batch_size):
            idx = torch.nonzero(mask[b], as_tuple=False).squeeze(1)  # masked positions in this row
            nm = idx.numel()
            if nm == 0:
                pos_to_unmask.append([])
                continue

            nk = min(k, nm)
            # Choose nk distinct masked positions uniformly
            perm = torch.randperm(nm, device=self.device)[:nk]
            pos_idx = idx[perm].tolist()                    # [nk]
            pos_to_unmask.append(pos_idx)
            # Get the distributions for those positions
            q_s = p_x0[b, pos_idx]                    # [nk]

            # Sample new tokens and write back
            new_tokens = 2*torch.bernoulli(q_s) - 1  # [nk]
            xt[b, pos_idx] = new_tokens

        return xt
    
    @torch.no_grad()
    def mask_and_sample(self, x0, mask_pos = None, T0 = None, decoding_strategy="fair"):
        """Mask T0 positions at random in each sample of x0 and sample the rest.
        
        Parameters
        ----------
        x0 : torch.Tensor
            Original data samples, shape [num_samples, L]
        T0 : int
            Number of positions to mask at the beginning.
        decoding_strategy : str
            One of "fair", "greedy", "verygreedy". See below for details.
        mask_pos : torch.Tensor
            1D tensor of positions to mask. If provided, T0 is ignored and these positions
            are masked in each sample.
        """
        num_samples, L = x0.shape
        xt0 = x0.clone()
        assert T0 is not None or mask_pos is not None, "Either T0 or mask_pos must be provided."
        assert T0 is None or mask_pos is None, "Only one of T0 or mask_pos must be provided."
        if T0 is not None:
            indx_seq = torch.cat([torch.randperm(L) for _ in range(num_samples)])
            indx_seq = indx_seq.view(num_samples, L)
            
        else: # mask_pos is provided
            assert mask_pos.ndim == 1
            T0 = mask_pos.numel()
            indx_seq = torch.cat([mask_pos[torch.randperm(T0)] for _ in range(num_samples)])
            indx_seq = indx_seq.view(num_samples, T0)
        for i in range(num_samples):
            xt0[i, indx_seq[i, :T0]] = self.mask_index
        
        history = []
        xt = xt0.clone()
        
        def eval(T):
            frac_masked = T / L
            frac_correct = (xt == x0).float().mean().item()
            frac_errors = ((xt != x0) & (xt != self.mask_index)).float().mean().item()
            history.append((frac_masked, frac_correct, frac_errors))
        
        eval(T0)
        for T in range(T0, 0, -1): # T = num of masked indices at the beginning of current step
            logits = self(xt) # [num_samples, L]
            if decoding_strategy == "verygreedy":
                # pick the most confident prediction among the masked indices
                logits[xt != self.mask_index] = 0  # ignore unmasked positions
                max_confidences, max_indices = logits.abs().max(dim=1) # [num_samples]
                to_unmask = max_indices
                assert to_unmask.shape == (num_samples,)
                assert (xt[torch.arange(num_samples), to_unmask] == self.mask_index).all() 
            else:
                to_unmask = indx_seq[:, T-1]
                
            logits = logits[torch.arange(num_samples), to_unmask] # [num_samples]
            px0 = torch.sigmoid(logits)
            if decoding_strategy == "fair":
                xt[torch.arange(num_samples), to_unmask] = 2*torch.bernoulli(px0) - 1
            elif decoding_strategy in ("greedy", "verygreedy"):
                xt[torch.arange(num_samples), to_unmask] = (px0 >= 0.5).float()*2 - 1
            else:
                raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
            eval(T-1)
            assert ((xt == self.mask_index).long().sum(dim=1) == T-1).all()
            
        return xt, history
    
    @torch.no_grad()
    def mask_and_sample_oneshot(self, x0, mask_pos=None, T0=None, decoding_strategy="fair"):
        """Mask T0 positions at random in each sample of x0 and sample all at once.
        
        In a one-shot experiment, all masked positions are revealed simultaneously
        based on the model's probability estimates at the masked positions.
        
        Parameters
        ----------
        x0 : torch.Tensor
            Original data samples, shape [num_samples, L]
        T0 : int
            Number of positions to mask at the beginning.
        decoding_strategy : str
            One of "fair" or "greedy". "fair" samples from the model's probabilities,
            "greedy" takes the argmax.
        mask_pos : torch.Tensor
            1D tensor of positions to mask. If provided, T0 is ignored and these positions
            are masked in each sample.
        
        Returns
        -------
        xt : torch.Tensor
            Sampled completions, shape [num_samples, L]
        metrics : dict
            Dictionary with keys: frac_masked, frac_correct, frac_errors
        """
        num_samples, L = x0.shape
        x0 = x0.to(self.device)
        xt0 = x0.clone()
        
        assert T0 is not None or mask_pos is not None, "Either T0 or mask_pos must be provided."
        assert T0 is None or mask_pos is None, "Only one of T0 or mask_pos must be provided."
        
        if T0 is not None:
            # Randomly select T0 positions to mask for each sample
            indx_seq = torch.cat([torch.randperm(L) for _ in range(num_samples)])
            indx_seq = indx_seq.view(num_samples, L)
        else:
            # mask_pos is provided
            assert mask_pos.ndim == 1
            T0 = mask_pos.numel()
            indx_seq = torch.cat([mask_pos[torch.randperm(T0)] for _ in range(num_samples)])
            indx_seq = indx_seq.view(num_samples, T0)
        
        # Mask the selected positions
        for i in range(num_samples):
            xt0[i, indx_seq[i, :T0]] = self.mask_index
        
        # Get model predictions for all positions
        logits = self(xt0)  # [num_samples, L]
        px0 = torch.sigmoid(logits)  # [num_samples, L]
        
        # Sample all masked positions at once
        xt = xt0.clone()
        mask = (xt == self.mask_index)  # [num_samples, L]
        
        if decoding_strategy == "fair":
            # Sample from the model's probability distribution
            samples = 2 * torch.bernoulli(px0) - 1  # [num_samples, L]
            xt[mask] = samples[mask]
        elif decoding_strategy in ("greedy", "verygreedy"):
            # Take the argmax
            samples = (px0 >= 0.5).float() * 2 - 1  # [num_samples, L]
            xt[mask] = samples[mask]
        else:
            raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
        
        # Compute metrics
        frac_masked = T0 / L
        frac_correct = (xt == x0).float().mean().item()
        frac_errors = ((xt != x0) & (xt != self.mask_index)).float().mean().item()
        
        metrics = {
            "frac_masked": frac_masked,
            "frac_correct": frac_correct,
            "frac_errors": frac_errors,
        }
        
        return xt, metrics
   
