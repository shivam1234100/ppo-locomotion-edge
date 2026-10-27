"""Running mean/std observation normalizer that lives inside the model.

Why an nn.Module and not a Gymnasium wrapper: the statistics are registered buffers, so they are
saved in the checkpoint's state_dict together with the weights and baked into the exported ONNX
graph. The deployed policy then consumes raw sensor readings, and the normalization used at serve
time can never drift from the one used in training (no train/serve skew).
"""

import torch
from torch import nn


class ObsNormalizer(nn.Module):
    def __init__(self, shape, enabled=True, clip=10.0, epsilon=1e-8):
        super().__init__()
        # Plain Python bool, not a buffer: it is fixed for a whole run (it is the ablation variable),
        # and at ONNX export the `if` in forward() is resolved once, so a disabled normalizer adds
        # no nodes to the graph. Checkpoints restore it from the saved config (obs_norm).
        self.enabled = enabled
        self.clip = clip
        self.epsilon = epsilon
        # Buffers, not Parameters: saved in state_dict and exported to ONNX, but invisible to the
        # optimizer, because they are estimated from data rather than learned by gradient descent.
        # float32 (Gymnasium's RunningMeanStd uses float64) so the exported graph is pure float32;
        # the extra rounding error is far below anything that matters for scaling inputs.
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        # Tiny pseudo-count (as in Gymnasium) so the first update never divides by zero, while the
        # initial mean=0 / var=1 guess carries almost no weight once real data arrives.
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()  # running statistics are bookkeeping, never part of the autograd graph
    def update(self, x):
        """Merge a batch of raw observations, shape (batch, *shape), into the running statistics."""
        if not self.enabled:
            return
        batch_mean = x.mean(dim=0)
        # Population variance (correction=0) to match np.var in Gymnasium. The unbiased estimator
        # would be NaN for a batch of one, which is exactly what num_envs=1 gives us every step.
        batch_var = x.var(dim=0, correction=0)
        batch_count = x.shape[0]

        # Parallel-variance combination (Chan et al.) of the old moments with the batch moments:
        # exact for any batch size and more stable than accumulating raw sums of x and x^2.
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count  # sum of squared deviations seen so far
        m_b = batch_var * batch_count  # sum of squared deviations in this batch
        # Correction term for the two groups having different means.
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count

        # copy_ writes in place, so the registered buffers stay the same tensor objects.
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot_count)
        self.count.copy_(tot_count)

    def forward(self, x):
        if not self.enabled:
            return x  # ablation arm: the network sees raw observations
        # epsilon guards against a zero-variance feature; clipping to [-10, 10] stops rare outliers
        # (e.g. a velocity spike as the robot falls) from saturating the tanh layers.
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.epsilon), -self.clip, self.clip)
