"""Profiling helper for ncsnpp_v2_5M to surface the heaviest compute blocks."""

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from sgmse.backbones.ncsnpp_v2 import NCSNpp_v2_5M


def build_complex_audio(batch_size=1, height=256, width=256):
    real = torch.randn(batch_size, 1, height, width)
    imag = torch.randn(batch_size, 1, height, width)
    complex_tensor = torch.view_as_complex(torch.stack([real, imag], dim=-1))
    return complex_tensor


def run_profile():
    model = NCSNpp_v2_5M().eval()

    x = build_complex_audio()
    y = build_complex_audio()
    t = torch.rand(1)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("forward pass"):
            with torch.no_grad():
                _ = model(x, y, t)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))


if __name__ == "__main__":
    run_profile()
