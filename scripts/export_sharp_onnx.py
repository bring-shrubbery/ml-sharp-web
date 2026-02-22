#!/usr/bin/env python3
"""Export Apple SHARP PyTorch weights to an ONNX wrapper for browser inference.

This script creates an ONNX model that wraps the upstream SHARP predictor and performs
an ONNX-friendly version of the NDC->metric Gaussian conversion. The exported model is
intended for use with the React/Bun web app in this repository.

Inputs (expected by the web app):
- image: float32 [1, 3, 1536, 1536], RGB in [0, 1]
- disparity_factor: float32 [1]          (f_px / image_width)
- f_px: float32 [1]                      (original image focal length in pixels)
- orig_width: float32 [1]
- orig_height: float32 [1]

Outputs:
- mean_vectors: float32 [1, N, 3]
- singular_values: float32 [1, N, 3]
- quaternions: float32 [1, N, 4]         (w, x, y, z)
- colors: float32 [1, N, 3]              (linear RGB)
- opacities: float32 [1, N]

Notes:
- Export success depends on your local PyTorch/ONNX versions and whether all ops (notably
  SVD) export cleanly for the SHARP graph.
- Runtime success in the browser depends on ONNX Runtime Web (WebGPU/WASM) operator support.
- The released Apple SHARP model weights are licensed separately for research-only use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sharp-repo",
        type=Path,
        required=True,
        help="Path to a local clone of https://github.com/apple/ml-sharp",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to SHARP .pt checkpoint (optional; downloads default if omitted).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/models/sharp_web_wrapper.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for export (cpu recommended for widest compatibility).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=20,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostics.",
    )
    return parser.parse_args()


def import_sharp_modules(sharp_repo: Path):
    src_path = sharp_repo / "src"
    if not src_path.exists():
        raise FileNotFoundError(
            f"Could not find {src_path}. Expected a local clone of apple/ml-sharp with a src/ directory."
        )
    sys.path.insert(0, str(src_path))

    import torch  # noqa: WPS433

    from sharp.cli.predict import DEFAULT_MODEL_URL  # noqa: WPS433
    from sharp.models import PredictorParams, create_predictor  # noqa: WPS433

    return torch, DEFAULT_MODEL_URL, PredictorParams, create_predictor


def rotation_matrices_from_quaternions(quaternions, torch):
    # quaternions: [..., 4] with (w, x, y, z)
    q = quaternions / torch.linalg.norm(quaternions, dim=-1, keepdim=True).clamp(min=1e-8)
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    m00 = ww + xx - yy - zz
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)

    m10 = 2.0 * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = 2.0 * (yz - wx)

    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = ww - xx - yy + zz

    row0 = torch.stack((m00, m01, m02), dim=-1)
    row1 = torch.stack((m10, m11, m12), dim=-1)
    row2 = torch.stack((m20, m21, m22), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def quaternions_from_rotation_matrices(rotations, torch):
    # rotations: [..., 3, 3] -> quaternions [..., 4] (w, x, y, z)
    m00 = rotations[..., 0, 0]
    m01 = rotations[..., 0, 1]
    m02 = rotations[..., 0, 2]
    m10 = rotations[..., 1, 0]
    m11 = rotations[..., 1, 1]
    m12 = rotations[..., 1, 2]
    m20 = rotations[..., 2, 0]
    m21 = rotations[..., 2, 1]
    m22 = rotations[..., 2, 2]

    eps = torch.as_tensor(1e-8, dtype=rotations.dtype, device=rotations.device)

    def safe_sqrt(x):
        return torch.sqrt(torch.clamp(x, min=0.0) + eps)

    def signed_unit(x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    qw = 0.5 * safe_sqrt(1.0 + m00 + m11 + m22)
    qx = 0.5 * safe_sqrt(1.0 + m00 - m11 - m22) * signed_unit(m21 - m12)
    qy = 0.5 * safe_sqrt(1.0 - m00 + m11 - m22) * signed_unit(m02 - m20)
    qz = 0.5 * safe_sqrt(1.0 - m00 - m11 + m22) * signed_unit(m10 - m01)

    q = torch.stack((qw, qx, qy, qz), dim=-1)
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)


def make_wrapper(predictor, torch):
    class SharpWebWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = predictor

        def _unproject_diagonal(self, gaussians, f_px, orig_width, orig_height):
            # In SHARP's predict_image(), the NDC->metric transform simplifies to diagonal scaling when
            # using centered intrinsics and identity extrinsics: diag(orig_w/(2*f), orig_h/(2*f), 1).
            sx = (orig_width / (2.0 * f_px)).view(-1, 1)
            sy = (orig_height / (2.0 * f_px)).view(-1, 1)
            sz = torch.ones_like(sx)
            scale_xyz = torch.stack((sx, sy, sz), dim=-1)  # [B,1,3]

            mean_vectors = gaussians.mean_vectors * scale_xyz

            rotations = rotation_matrices_from_quaternions(gaussians.quaternions, torch)
            variances = gaussians.singular_values.square()
            cov = rotations @ torch.diag_embed(variances) @ rotations.transpose(-1, -2)

            d = torch.zeros(
                (scale_xyz.shape[0], 1, 3, 3),
                dtype=mean_vectors.dtype,
                device=mean_vectors.device,
            )
            d[..., 0, 0] = sx
            d[..., 1, 1] = sy
            d[..., 2, 2] = 1.0

            cov_metric = d @ cov @ d.transpose(-1, -2)
            u, s, _ = torch.linalg.svd(cov_metric)

            det_u = torch.linalg.det(u)
            flip = (det_u < 0).to(u.dtype).unsqueeze(-1)
            u_last = u[..., :, 2] * (1.0 - 2.0 * flip)
            rotations_fixed = torch.cat((u[..., :, 0:2], u_last.unsqueeze(-1)), dim=-1)

            singular_values = torch.sqrt(torch.clamp(s, min=0.0))
            quaternions = quaternions_from_rotation_matrices(rotations_fixed, torch)

            return mean_vectors, singular_values, quaternions

        def forward(self, image, disparity_factor, f_px, orig_width, orig_height):
            gaussians = self.predictor(image, disparity_factor)
            mean_vectors, singular_values, quaternions = self._unproject_diagonal(
                gaussians, f_px, orig_width, orig_height
            )
            return (
                mean_vectors,
                singular_values,
                quaternions,
                gaussians.colors,
                gaussians.opacities,
            )

    return SharpWebWrapper()


def load_predictor(torch, create_predictor, PredictorParams, checkpoint_path, default_url, device, verbose):
    predictor = create_predictor(PredictorParams())
    if checkpoint_path is None:
        if verbose:
            print(f"Downloading checkpoint from {default_url}")
        state_dict = torch.hub.load_state_dict_from_url(default_url, progress=True)
    else:
        if verbose:
            print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)

    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(device)
    return predictor


def export_onnx():
    args = parse_args()
    sharp_repo = args.sharp_repo.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else None

    torch, default_url, PredictorParams, create_predictor = import_sharp_modules(sharp_repo)

    device = torch.device(args.device)
    predictor = load_predictor(
        torch=torch,
        create_predictor=create_predictor,
        PredictorParams=PredictorParams,
        checkpoint_path=checkpoint_path,
        default_url=default_url,
        device=device,
        verbose=args.verbose,
    )

    wrapper = make_wrapper(predictor, torch).eval().to(device)

    dummy_image = torch.rand((1, 3, 1536, 1536), dtype=torch.float32, device=device)
    dummy_disparity_factor = torch.tensor([1.0], dtype=torch.float32, device=device)
    dummy_f_px = torch.tensor([1024.0], dtype=torch.float32, device=device)
    dummy_orig_width = torch.tensor([1024.0], dtype=torch.float32, device=device)
    dummy_orig_height = torch.tensor([768.0], dtype=torch.float32, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Exporting ONNX to {output_path}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (
                dummy_image,
                dummy_disparity_factor,
                dummy_f_px,
                dummy_orig_width,
                dummy_orig_height,
            ),
            str(output_path),
            export_params=True,
            do_constant_folding=True,
            opset_version=args.opset,
            input_names=["image", "disparity_factor", "f_px", "orig_width", "orig_height"],
            output_names=["mean_vectors", "singular_values", "quaternions", "colors", "opacities"],
        )

    print(f"Wrote {output_path}")
    print(
        "Next: place the ONNX file under public/models/ (or load it via the UI), then run `bun dev`."
    )


if __name__ == "__main__":
    export_onnx()
