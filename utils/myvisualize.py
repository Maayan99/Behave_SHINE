import math
from typing import Any, Dict, Optional, Tuple, List

import torch
import matplotlib.pyplot as plt
import seaborn as sns


def _to_2d_for_plot(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float().cpu()
    if x.ndim == 1:
        return x.unsqueeze(0)
    if x.ndim == 2:
        return x
    return x.reshape(-1, x.shape[-1])


def _downsample_2d(x2d: torch.Tensor, max_hw: int = 512) -> torch.Tensor:
    h, w = x2d.shape
    if h <= max_hw and w <= max_hw:
        return x2d
    sh = max(1, math.ceil(h / max_hw))
    sw = max(1, math.ceil(w / max_hw))
    return x2d[::sh, ::sw]


def _iter_lora_leaves(loradict: Dict[str, Any]) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    leaves: List[Tuple[str, Dict[str, torch.Tensor]]] = []

    def rec(node: Any, path: str):
        if isinstance(node, dict):
            if "A" in node and "B" in node:
                leaves.append((path, node))
                return
            for k, v in node.items():
                rec(v, f"{path}.{k}" if path else str(k))

    rec(loradict, "")
    return leaves


def _sanitize_filename(s: str) -> str:
    # safe-ish filename from path
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)


def _sns_heatmap(ax, mat2d: torch.Tensor, title: str):
    # seaborn handles scaling nicely; turn off tick labels for readability
    sns.heatmap(mat2d, ax=ax, cbar=True, xticklabels=False, yticklabels=False)
    ax.set_title(title)


# @torch.no_grad()
# def visualize_loradict_to_files(
#     loradict: Dict[str, Any],
#     out_dir: str,
#     layer: Optional[int] = None,
#     only: Optional[str] = None,
#     batch_index: int = 0,
#     max_hw: int = 10000000,
#     dpi: int = 2000,
# ):
#     """
#     Save seaborn heatmaps for A, B, and A@B (and C if present) for each LoRA leaf.

#     Args:
#       loradict: nested dict with leaves like {"A":..., "B":..., "C":...}
#       out_dir: directory to write images into
#       layer: restrict to one layer index (top-level int key) if provided
#       only: substring filter on leaf path (e.g. "attention.q", "mlp.gate")
#       batch_index: which LoRA batch entry to visualize (leading dim of A/B/C)
#       max_hw: downsample limit for heatmaps
#       dpi: figure save DPI
#     """
#     import os
#     os.makedirs(out_dir, exist_ok=True)

#     # Restrict to a single layer if requested
#         if layer is not None:
#         if layer not in loradict:
#             raise KeyError(f"layer={layer} not found in loradict keys={list(loradict.keys())[:10]}...")
#         loradict_view = {layer: loradict[layer]}
#     else:
#         loradict_view = loradict

#     leaves = _iter_lora_leaves(loradict_view)
#     if only is not None:
#         leaves = [(p, d) for (p, d) in leaves if only in p]

#     if not leaves:
#         raise ValueError("No LoRA leaves found to visualize (check `layer` / `only`).")

#     saved_paths = []

#     for path, leaf in leaves:
#         print(f"Visualizing: {path}")
#         A = leaf["A"]
#         B = leaf["B"]
#         C = leaf.get("C", None)

#         # Select batch entry (A: [B,in,r], B: [B,r,out], C: [B,out])
#         if A.ndim >= 3:
#             if batch_index >= A.shape[0]:
#                 raise IndexError(f"batch_index={batch_index} out of range for A.shape[0]={A.shape[0]}")
#             A0 = A[batch_index]
#         else:
#             A0 = A

#         if B.ndim >= 3:
#             if batch_index >= B.shape[0]:
#                 raise IndexError(f"batch_index={batch_index} out of range for B.shape[0]={B.shape[0]}")
#             B0 = B[batch_index]
#         else:
#             B0 = B

#         # Low-rank update ΔW
#         dW = A0 @ B0  # (in, out)

#         # Prepare 2D + downsample
#         A2 = _downsample_2d(_to_2d_for_plot(A0), max_hw=max_hw)
#         B2 = _downsample_2d(_to_2d_for_plot(B0), max_hw=max_hw)
#         dW2 = _downsample_2d(_to_2d_for_plot(dW), max_hw=max_hw)

#         has_C = C is not None
#         ncols = 4 if has_C else 3
#         fig, axes = plt.subplots(1, ncols, figsize=(24 * ncols, 21))

#         if ncols == 3:
#             axA, axB, axDW = axes
#         else:
#             axA, axB, axDW, axC = axes

#         _sns_heatmap(axA, A2, f"{path}\nA (in×r)  {tuple(A0.shape)} -> {tuple(A2.shape)}")
#         _sns_heatmap(axB, B2, f"{path}\nB (r×out) {tuple(B0.shape)} -> {tuple(B2.shape)}")
#         _sns_heatmap(axDW, dW2, f"{path}\nA@B (ΔW)  {tuple(dW.shape)} -> {tuple(dW2.shape)}")

#         if has_C:
#             C0 = C[batch_index] if C.ndim >= 2 else C
#             C2 = _downsample_2d(_to_2d_for_plot(C0), max_hw=max_hw)
#             _sns_heatmap(axC, C2, f"{path}\nC (bias)   {tuple(C0.shape)} -> {tuple(C2.shape)}")

#         fig.tight_layout()

#         fname = f"{_sanitize_filename(path)}__b{batch_index}.png"
#         fpath = os.path.join(out_dir, fname)
#         fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
#         print(f"Saved to: {fpath}")
#         plt.close(fig)

#         saved_paths.append(fpath)

#     return saved_paths


# def summarize_loradict_shapes(loradict: Dict[str, Any], layer: Optional[int] = None):
#     if layer is not None:
#         loradict = {layer: loradict[layer]}
#     leaves = _iter_lora_leaves(loradict)
#     for path, leaf in leaves:
#         A = leaf["A"]
#         B = leaf["B"]
#         C = leaf.get("C", None)
#         print(
#             f"{path:40s}  A={tuple(A.shape)}  B={tuple(B.shape)}  "
#             f"C={tuple(C.shape) if C is not None else None}"
#         )


@torch.no_grad()
def visualize_loradict_to_files(
    loradict: Dict[str, Any],
    out_dir: str,
    layer: Optional[int] = None,
    only: Optional[str] = None,
    batch_index: int = 0,
    crop_in: int = 128,    # rows for A/dW (in dimension)
    crop_out: int = 128,   # cols for B/dW (out dimension)
    crop_r: int = 4,       # LoRA rank
    dpi: int = 300,
):
    """
    Save heatmaps for A, B, and A@B (and C if present) for each LoRA leaf.

    Changes vs original:
      (1) Layout: A left, B top, C at bottom-right (aligned with A horizontally, B vertically).
          ΔW placed at top-left.
      (2) Crop visualization to:
          A: crop_in x crop_r
          B: crop_r x crop_out
          ΔW: crop_in x crop_out
          C: first crop_out entries (shown as 1 x crop_out)
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Restrict to a single layer if requested
    if layer is not None:
        if layer not in loradict:
            raise KeyError(f"layer={layer} not found in loradict keys={list(loradict.keys())[:10]}...")
        loradict_view = {layer: loradict[layer]}
    else:
        loradict_view = loradict

    leaves = _iter_lora_leaves(loradict_view)
    if only is not None:
        leaves = [(p, d) for (p, d) in leaves if only in p]

    if not leaves:
        raise ValueError("No LoRA leaves found to visualize (check `layer` / `only`).")

    def _select_batch(t: torch.Tensor) -> torch.Tensor:
        if t.ndim >= 3:
            if batch_index >= t.shape[0]:
                raise IndexError(f"batch_index={batch_index} out of range for shape[0]={t.shape[0]}")
            return t[batch_index]
        return t

    def _as_2d(t: torch.Tensor) -> torch.Tensor:
        t = t.detach().float().cpu()
        if t.ndim == 1:
            return t.unsqueeze(0)  # 1 x N
        if t.ndim == 2:
            return t
        return t.reshape(-1, t.shape[-1])

    def _crop2d(t2d: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return t2d[: min(h, t2d.shape[0]), : min(w, t2d.shape[1])]

    saved_paths = []

    for path, leaf in leaves:
        print(f"Visualizing: {path}")
        A = _select_batch(leaf["A"])
        B = _select_batch(leaf["B"])
        C = leaf.get("C", None)
        C0 = _select_batch(C) if C is not None else None

        A2 = _as_2d(A)  # (in, r) expected
        B2 = _as_2d(B)  # (r, out) expected

        # Crop A and B before multiplication so ΔW is also cropped
        A2c = _crop2d(A2, crop_in, crop_r)   # 128x4
        B2c = _crop2d(B2, crop_r, crop_out)  # 4x128

        dWc = A2c @ B2c                      # 128x128

        has_C = C0 is not None

        # --- Layout: 2x2 grid ---
        # [0,0] ΔW   [0,1] B
        # [1,0] A    [1,1] C (if present; else blank)
        fig, axes = plt.subplots(2, 2, figsize=(32, 28))

        axDW = axes[0, 0]
        axB  = axes[0, 1]
        axA  = axes[1, 0]
        axC  = axes[1, 1]

        _sns_heatmap(axA, A2c, f"{path}\nA crop (in×r) {tuple(A2.shape)} -> {tuple(A2c.shape)}")
        _sns_heatmap(axB, B2c, f"{path}\nB crop (r×out) {tuple(B2.shape)} -> {tuple(B2c.shape)}")
        _sns_heatmap(axDW, dWc, f"{path}\nA@B crop (ΔW) -> {tuple(dWc.shape)}")

        if has_C:
            C2 = _as_2d(C0)
            # If C is (out,) or (something, out), show first crop_out along last dim as 1 x crop_out
            if C2.shape[0] == 1:
                Cc = _crop2d(C2, 1, crop_out)
            else:
                # If it came in as (out, ?) unexpectedly, just take first row after reshape
                Cc = _crop2d(C2[:1, :], 1, crop_out)
            _sns_heatmap(axC, Cc, f"{path}\nC crop (bias) {tuple(C2.shape)} -> {tuple(Cc.shape)}")
        else:
            axC.axis("off")
            axC.set_title(f"{path}\nC (bias) not present")

        fig.tight_layout()

        fname = f"{_sanitize_filename(path)}__b{batch_index}.png"
        fpath = os.path.join(out_dir, fname)
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        print(f"Saved to: {fpath}")
        plt.close(fig)

        saved_paths.append(fpath)

    return saved_paths

