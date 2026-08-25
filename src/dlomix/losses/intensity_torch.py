import numpy as np
import torch
import torch.nn.functional as F


def masked_spectral_distance(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the masked spectral distance between true and predicted intensity vectors.
    The masked spectral distance is a metric for comparing the similarity between two intensity vectors.

    Masked, normalized spectral angles between true and pred vectors

    > arccos(1*1 + 0*0) = 0 -> SL = 0 -> high correlation

    > arccos(0*1 + 1*0) = pi/2 -> SL = 1 -> low correlation

    Parameters
    ----------
    y_true : torch.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : torch.Tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    torch.Tensor
        A tensor containing the masked spectral distance between `y_true` and `y_pred`.

    """

    # To avoid numerical instability during training on GPUs,
    # we add a fuzzing constant epsilon of 1×10−7 to all vectors
    epsilon = 1e-7

    # Masking: we multiply values by (true + 1) because then the peaks that cannot
    # be there (and have value of -1 as explained above) won't be considered
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    # L2 norm
    # along last axis / dimension of the tensor
    true_norm = F.normalize(true_masked, p=2, dim=-1)
    pred_norm = F.normalize(pred_masked, p=2, dim=-1)

    # Spectral Angle (SA) calculation
    # (from the definition below, it is clear that ions with higher intensities
    #  will always have a higher contribution)
    product = (pred_norm * true_norm).sum(dim=-1)
    product = torch.clamp(product, -1.0 + epsilon, 1.0 - epsilon)
    arccos = torch.arccos(product)
    batch_losses = 2 * arccos / np.pi

    return batch_losses.mean()


def masked_pearson_correlation_distance(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the masked Pearson correlation distance between true and predicted intensity vectors.
    The masked Pearson correlation distance is a metric for comparing the similarity between two intensity vectors,
    taking into account only the non-negative values in the true values tensor (which represent valid peaks).

    Parameters
    ----------
    y_true : torch.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_pred : torch.Tensor
        A tensor containing the predicted values, with the same shape as `y_true`.

    Returns
    -------
    torch.Tensor
        A tensor containing the masked Pearson correlation distance between `y_true` and `y_pred`.

    """

    epsilon = 1e-7

    # Masking: we multiply values by (true + 1) because then the peaks that cannot
    # be there (and have value of -1 as explained above) won't be considered
    pred_masked = ((y_true + 1) * y_pred) / (y_true + 1 + epsilon)
    true_masked = ((y_true + 1) * y_true) / (y_true + 1 + epsilon)

    mx = true_masked.mean()
    my = pred_masked.mean()
    xm, ym = true_masked - mx, pred_masked - my
    r_num = (xm * ym).mean()
    r_den = xm.std(unbiased=False) * ym.std(unbiased=False)

    return 1 - (r_num / r_den)


def _infer_fragments_per_cleavage(
    y_true: torch.Tensor, encoded_sequence: torch.Tensor, has_termini: bool
) -> int:
    """Infer how many ion channels are stored per peptide cleavage.

    Prosit-style intensity vectors flatten all predicted fragment-ion channels
    into one axis. For a peptide with L amino-acid residues, there are L - 1
    internal cleavages. With max_seq_len=32 and terminal tokens enabled, the
    largest peptide has 30 residues, therefore 29 cleavages. A 174-wide target
    tensor is then interpreted as 29 cleavages * 6 b/y charge channels.
    """
    # Encoded batches may include synthetic N- and C-terminal tokens. They are
    # sequence context tokens, not amino-acid residues that can create b/y cuts.
    terminal_count = 2 if has_termini else 0

    # encoded_sequence.shape[-1] is the padded sequence width, e.g. max_seq_len.
    # Subtract terminal tokens to get the maximum residue count represented by
    # this tensor shape. The max(..., 1) avoids zero in malformed tiny examples.
    max_residue_count = max(int(encoded_sequence.shape[-1]) - terminal_count, 1)

    # A peptide with L residues has L - 1 internal backbone cleavages. Each
    # cleavage can produce several ion channels, such as b/y ions for charges
    # 1, 2, and 3. Again clamp to at least one to keep diagnostics well-defined.
    max_cleavages = max(max_residue_count - 1, 1)

    # If the flattened target width divides cleanly by the number of cleavages,
    # the quotient is the number of ion channels stored for each cleavage.
    if y_true.shape[-1] % max_cleavages == 0:
        return int(y_true.shape[-1] // max_cleavages)

    # Prosit intensity models normally predict six channels per cleavage:
    # b1, b2, b3, y1, y2, y3. Keep this as a conservative fallback for callers
    # that provide tensors with a non-standard padded shape.
    return 6


def _possible_fragment_mask(
    y_true: torch.Tensor,
    encoded_sequence: torch.Tensor,
    fragments_per_cleavage=None,
    has_termini: bool = True,
) -> torch.Tensor:
    """Return a boolean mask for fragment-ion positions possible for each peptide.

    Parameters
    ----------
    y_true : torch.Tensor
        Intensity target tensor with shape `(batch_size, flattened_ion_count)`.
        The final axis is ordered by peptide cleavage and ion channel. For a
        peptide shorter than the configured maximum length, trailing positions in
        this axis correspond to cleavages that cannot exist.
    encoded_sequence : torch.Tensor
        Padded integer-encoded peptide sequences with shape
        `(batch_size, max_seq_len)`. Non-zero entries are real sequence/context
        tokens. Zero entries are padding.
    fragments_per_cleavage : int, optional
        Number of flattened target positions used for one peptide cleavage. In
        standard Prosit intensity this is 6: b/y ions for charges 1, 2, and 3.
        If omitted, infer it from the target width and padded sequence width.
    has_termini : bool, optional
        Whether `encoded_sequence` includes N- and C-terminal tokens. When true,
        those two tokens are excluded from residue-length and cleavage counts.

    Returns
    -------
    torch.Tensor
        Boolean tensor with the same shape as `y_true`; true means the position
        belongs to a theoretical b/y fragment for that peptide length.
    """
    if y_true.ndim != 2:
        raise ValueError(
            f"Expected y_true with shape (batch, ions), got {tuple(y_true.shape)}"
        )
    if encoded_sequence.ndim != 2:
        raise ValueError(
            "Expected encoded_sequence with shape (batch, sequence), got "
            f"{tuple(encoded_sequence.shape)}"
        )
    if y_true.shape[0] != encoded_sequence.shape[0]:
        raise ValueError(
            "Batch size mismatch between y_true and encoded_sequence: "
            f"{y_true.shape[0]} != {encoded_sequence.shape[0]}"
        )

    # Keep all generated masks and index tensors on the target device. The
    # earlier CPU-created index tensor is a common cause of CUDA
    # advanced-indexing failures once the batch lives on GPU.
    encoded_sequence = encoded_sequence.to(device=y_true.device)

    # Determine how many flattened ion positions correspond to one cleavage.
    if fragments_per_cleavage is None:
        fragments_per_cleavage = _infer_fragments_per_cleavage(
            y_true, encoded_sequence, has_termini
        )

    # Count non-padding tokens per peptide. If terminal tokens are present,
    # subtract them because they do not represent residues or cleavage sites.
    terminal_count = 2 if has_termini else 0
    residue_counts = (encoded_sequence != 0).sum(dim=1) - terminal_count
    residue_counts = residue_counts.clamp(min=0)

    # Convert residue counts to flattened ion counts. A peptide with L residues
    # has L - 1 cleavages; each cleavage contributes fragments_per_cleavage
    # contiguous positions in the flattened intensity vector.
    valid_counts = (residue_counts - 1).clamp(min=0) * int(fragments_per_cleavage)
    valid_counts = valid_counts.clamp(max=y_true.shape[-1])

    # Compare every flattened ion position against each peptide's valid count.
    # Broadcasting gives shape (batch_size, flattened_ion_count).
    fragment_positions = torch.arange(y_true.shape[-1], device=y_true.device)
    return fragment_positions.unsqueeze(0) < valid_counts.unsqueeze(1)


def gaussian_nll(
    y_true: torch.Tensor,
    y_log_mean_pred: torch.Tensor,
    y_log_var_pred: torch.Tensor,
    y_presence_pred: torch.Tensor,
    encoded_sequence: torch.Tensor,
    bce_weight: float,
    nll_weight: float,
    fragments_per_cleavage=None,
    has_termini: bool = True,
) -> torch.Tensor:
    """
    Calculates a zero-inflated log-normal loss.

    This implements the decomposed mixture objective:

        L_mix = BCE(t_k, logit_k) + sum_{k: y_k > 0} GaussianNLL_k

    where the BCE term is evaluated for valid theoretical ions, `t_k = 1` means
    the fragment is present (`y_k > 0`), and the Gaussian term is evaluated only
    for present fragments in log-intensity space.

    Intensities with the sentinel value -1 are impossible ions and are ignored.
    Valid zero-intensity ions still contribute to the BCE absence term.

    Parameters
    ----------
    y_true : torch.Tensor
        A tensor containing the true values, with shape `(batch_size, num_values)`.
    y_log_mean_pred : torch.Tensor
        A tensor containing predicted mean log-intensities, with the same shape
        as `y_true`.
    y_log_var_pred : torch.Tensor
        A tensor containing predicted log variances, with the same shape as
        `y_true`.
    y_presence_pred : torch.Tensor
        A tensor containing predicted presence logits, with the same shape as
        `y_true`.
    encoded_sequence : torch.Tensor
        Tensor containing the number encoded sequence. Shape is equal to
        `(batch_size, max_seq_len)`.
    bce_weight : float
        The weight for the BCE loss
    nll_weight : float
        The weight for the Gaussian NLL loss
    fragments_per_cleavage : int, optional
        Number of fragment-ion channels predicted per peptide cleavage. Inferred
        from tensor shape when omitted.
    has_termini : bool, optional
        Whether encoded sequences include N- and C-terminal tokens.

    Returns
    -------
    torch.Tensor
        Mean per-valid-fragment mixture negative log likelihood.

    """
    epsilon = 1e-7

    # Autocast may run the model in bf16/fp16. Keep the probabilistic loss in
    # fp32; tiny variances and log-intensity errors are exactly where reduced
    # precision can turn a large finite loss into inf/NaN.
    y_true = y_true.float()
    y_log_mean_pred = y_log_mean_pred.float()
    y_log_var_pred = y_log_var_pred.float()
    y_presence_pred = y_presence_pred.float()

    # possible: positions that can exist for each peptide length.
    # y_true >= 0: remove -1 sentinels for impossible/unannotated ions. Combining
    # the two gives the domain k in b,y over which the mixture likelihood is
    # defined for this batch.
    possible = _possible_fragment_mask(
        y_true,
        encoded_sequence,
        fragments_per_cleavage=fragments_per_cleavage,
        has_termini=has_termini,
    )
    valid = possible & (y_true >= 0)
    present = valid & (y_true > 0)

    if not torch.any(valid):
        raise ValueError(
            "No valid fragment ions remain after applying peptide-length and -1 masks. "
            "Check encoded_sequence, has_termini, fragments_per_cleavage, and target shape."
        )

    # BCEWithLogitsLoss implements:
    #   -t_k * log(sigmoid(logit_k)) - (1 - t_k) * log(1 - sigmoid(logit_k))
    # which is exactly the Bernoulli part of the pasted mixture objective:
    #   -log p_k for y_k > 0, and -log(1 - p_k) for y_k = 0.
    # We use reduction="sum" first so the final normalization is controlled by
    # the same valid-fragment count as the full mixture loss.
    presence_target = present.to(dtype=y_presence_pred.dtype)
    presence_loss = F.binary_cross_entropy_with_logits(
        y_presence_pred[valid], presence_target[valid], reduction="sum"
    )

    if not torch.isfinite(y_log_mean_pred[valid]).all():
        raise ValueError("Non-finite predicted log-intensity mean in valid fragments.")
    if not torch.isfinite(y_log_var_pred[valid]).all():
        raise ValueError("Non-finite predicted log variance in valid fragments.")
    if not torch.isfinite(y_presence_pred[valid]).all():
        raise ValueError("Non-finite predicted presence logits in valid fragments.")

    if torch.any(present):
        # The Gaussian term in the derivation is over log(y_k + eps), not raw
        # intensity. Therefore the model's mean head is interpreted as mu_k in
        # log-intensity space, and the target passed to GaussianNLL is log y.
        log_target = torch.log(y_true[present].to(dtype=y_log_mean_pred.dtype) + epsilon)

        # Use the standard log-variance parameterization s = log(sigma^2):
        #   GaussianNLL_k = 0.5 * (exp(-s_k) * (log_y_k - mu_k)^2
        #                         + s_k + log(2*pi))
        # This is algebraically the same Gaussian NLL as PyTorch's
        # gaussian_nll_loss with var=sigma^2, but avoids requiring the network
        # to output a strictly positive variance directly. Keep this unclamped
        # for the baseline experiment so out-of-range log variances still get
        # gradients from the likelihood instead of hitting clamp dead zones.
        log_var = y_log_var_pred[present]
        squared_error = torch.square(log_target - y_log_mean_pred[present])
        intensity_loss = 0.5 * (
            torch.exp(-log_var) * squared_error
            + log_var
            + torch.log(
                torch.tensor(2.0 * np.pi, device=log_var.device, dtype=log_var.dtype)
            )
        )
        intensity_loss = intensity_loss.sum()
    else:
        # This is not a data-repair guard: if all valid ions are observed as zero,
        # the derivation's sum over {k: y_k > 0} is an empty sum, i.e. zero. The
        # batch still trains through the Bernoulli absence terms above.
        intensity_loss = y_log_mean_pred.sum() * 0.0

    total_loss = (bce_weight * presence_loss) + (nll_weight * intensity_loss)
    normalizer = valid.sum().to(dtype=total_loss.dtype)
    return total_loss / normalizer
