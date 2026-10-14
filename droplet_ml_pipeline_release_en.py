"""
Microscope droplet image analysis and machine-learning pipeline.

What it does
------------
1. Select an image root folder with a GUI dialog.
2. Read all images in the selected folder and its direct subfolders.
   Each subfolder name is used as the experimental group label.
3. Background correction, denoising, droplet segmentation.
4. Extract image-level features:
   phi, Cmax, CV_phi, droplet count, mean spacing, area statistics.
5. Extract connected-region features:
   area, circularity, eccentricity, solidity, spacing, intensity statistics.
6. Estimate one global standard droplet area A0 from all valid groups.
7. Fit one two-component GMM to x = ln(Ai / A0). The component intersection is
   reported as the unified q_ML threshold.
8. Bootstrap q_ML by resampling whole images and report a 95% confidence
    interval.
9. Write compact model-facing tables and figures:
    dynamics_input, image_results, sample_summary, model_parameters, GMM area
    density, area distributions, image-level parameter bubbles, and numeric
    correlation heatmap.
10. Optional single-image recognition mode saves only the corrected grayscale
    image, binary mask, and segmentation overlay.

Image calibration
-----------------
The field of view is assumed to be 1260 x 940 micrometers. Pixel size is
computed from each image's pixel width and height.
"""

from __future__ import annotations

import math
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import Canvas, Entry, IntVar, Scrollbar, StringVar, Tk, filedialog, messagebox, simpledialog, ttk

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageTk
from matplotlib import font_manager

warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", message=r".*Could not find the number of physical cores.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module=r"joblib\.externals\.loky\.backend\.context")
warnings.filterwarnings("ignore", category=UserWarning, module=r".*loky.*")

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


_ORIGINAL_TK_VARIABLE_DEL = tk.Variable.__del__


def safe_tk_variable_del(self) -> None:
    try:
        _ORIGINAL_TK_VARIABLE_DEL(self)
    except RuntimeError as exc:
        if "main thread is not in main loop" not in str(exc):
            raise


tk.Variable.__del__ = safe_tk_variable_del


def configure_matplotlib_fonts() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    else:
        warnings.filterwarnings("ignore", message=r"Glyph .* missing from font.*", category=UserWarning)
    plt.rcParams["axes.unicode_minus"] = False


configure_matplotlib_fonts()


# ----------------------------- User parameters -----------------------------

FIELD_WIDTH_UM = 1260.0
FIELD_HEIGHT_UM = 940.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Small regions and dust are removed below this area.
MIN_DROPLET_AREA_UM2 = 80.0

PREVIEW_MAX_SIDE = 650
USE_GPU_ACCELERATION = True
BATCH_PARALLEL_PROCESSING = True
MAX_BATCH_WORKERS = max(1, min(10, (os.cpu_count() or 2) - 1))

# Regions touching the image border are often partial droplets. Keep them in
# segmentation overlays but exclude them from A0/q estimation by default.
EXCLUDE_BORDER_FOR_A0_AND_Q = True

# Images with too few detected droplets are usually failed segmentations, such
# as the whole field being merged into one connected component.
MIN_VALID_DROPLETS_PER_IMAGE = 3

# Local phi grid for Cmax and CV_phi.
PHI_GRID_ROWS = 6
PHI_GRID_COLS = 8

# Bootstrap count for q_ML confidence interval. Bootstrap units are images.
N_BOOTSTRAP = 500
RANDOM_SEED = 42
DEFAULT_PARAMS_FILE = Path(__file__).with_name("droplet_segmentation_params_release.json")


@dataclass
class SegmentationResult:
    image_rgb: np.ndarray
    gray_corrected: np.ndarray
    mask: np.ndarray
    labels: np.ndarray


@dataclass
class SegmentationParams:
    gray_threshold_offset: int = 0
    min_droplet_area_um2: float = MIN_DROPLET_AREA_UM2

    @classmethod
    def from_dict(cls, data: dict) -> "SegmentationParams":
        params = cls()
        if "gray_threshold_offset" in data:
            params.gray_threshold_offset = int(data["gray_threshold_offset"])
        if "min_droplet_area_um2" in data:
            params.min_droplet_area_um2 = float(data["min_droplet_area_um2"])
        return params


@dataclass
class GroupFolder:
    group: str
    folder: Path


def load_segmentation_params(path: Path = DEFAULT_PARAMS_FILE) -> SegmentationParams:
    if path.exists():
        return SegmentationParams.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return SegmentationParams()


def save_segmentation_params(params: SegmentationParams, path: Path = DEFAULT_PARAMS_FILE) -> None:
    path.write_text(json.dumps(asdict(params), ensure_ascii=False, indent=2), encoding="utf-8")


_CUDA_DEVICE_COUNT: int | None = None


def cuda_device_count() -> int:
    global _CUDA_DEVICE_COUNT
    if _CUDA_DEVICE_COUNT is not None:
        return _CUDA_DEVICE_COUNT
    try:
        _CUDA_DEVICE_COUNT = int(cv2.cuda.getCudaEnabledDeviceCount()) if hasattr(cv2, "cuda") else 0
    except Exception:
        _CUDA_DEVICE_COUNT = 0
    return _CUDA_DEVICE_COUNT


def gpu_acceleration_available() -> bool:
    return bool(USE_GPU_ACCELERATION and cuda_device_count() > 0)


def gpu_status_text() -> str:
    if gpu_acceleration_available():
        return f"CUDA GPU acceleration enabled ({cuda_device_count()} device(s))"
    return "CUDA GPU acceleration unavailable; using CPU"


def cv_cuda_type_for(image: np.ndarray) -> int:
    if image.dtype != np.uint8:
        raise ValueError("CUDA helper only supports uint8 images")
    if image.ndim == 2:
        return cv2.CV_8UC1
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.CV_8UC3
    raise ValueError(f"Unsupported image shape for CUDA helper: {image.shape}")


def gaussian_blur_u8(image: np.ndarray, ksize: tuple[int, int], sigma: float = 0) -> np.ndarray:
    if not gpu_acceleration_available():
        return cv2.GaussianBlur(image, ksize, sigma)
    try:
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(np.ascontiguousarray(image))
        image_type = cv_cuda_type_for(image)
        gaussian_filter = cv2.cuda.createGaussianFilter(image_type, image_type, ksize, sigma)
        return gaussian_filter.apply(gpu_src).download()
    except Exception:
        return cv2.GaussianBlur(image, ksize, sigma)


def bilateral_denoise_u8(image: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    if not gpu_acceleration_available():
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    try:
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(np.ascontiguousarray(image))
        bilateral_filter = cv2.cuda.createBilateralFilter(cv2.CV_8UC1, cv2.CV_8UC1, d, sigma_color, sigma_space)
        return bilateral_filter.apply(gpu_src).download()
    except Exception:
        return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def choose_root_folder() -> Path:
    folder_args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if folder_args:
        folder = Path(folder_args[0]).expanduser().resolve()
        if folder.exists() and folder.is_dir():
            return folder

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select the folder containing microscope images")
    root.destroy()
    if not folder:
        raise SystemExit("No folder selected. Program terminated.")
    return Path(folder)


def collect_images(root: Path) -> list[Path]:
    paths = [
        p
        for p in root.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith("droplet_ml_results_") for part in p.relative_to(root).parts)
    ]
    return sorted(paths, key=lambda p: str(p).lower())


def collect_group_images(group_folders: list[GroupFolder]) -> list[tuple[Path, str, Path, Path]]:
    records = []
    for group_folder in group_folders:
        for image_path in collect_images(group_folder.folder):
            rel_stem = image_path.relative_to(group_folder.folder).with_suffix("")
            output_stem = Path(safe_path_part(group_folder.group)) / rel_stem
            records.append((image_path, group_folder.group, group_folder.folder, output_stem))
    return sorted(records, key=lambda item: (item[1].lower(), str(item[0]).lower()))


def safe_path_part(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text.strip())
    return cleaned or "group"


def common_output_root(group_folders: list[GroupFolder]) -> Path:
    if not group_folders:
        return Path.cwd()
    if len(group_folders) == 1:
        return group_folders[0].folder
    common = Path(os.path.commonpath([str(group_folder.folder) for group_folder in group_folders]))
    return common if common.exists() and common.is_dir() else group_folders[0].folder.parent


def infer_group(image_path: Path, root: Path) -> str:
    rel = image_path.relative_to(root)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return root.name


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def smooth_mask_boundary(mask: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Round off pixel-level jaggedness on the mask boundary.

    The saturation channel has some per-pixel measurement noise, which after
    thresholding shows up as a serrated/zig-zag boundary rather than the
    physically smooth curve a droplet or coalesced cluster actually has.
    Blurring the binary mask and re-thresholding at the midpoint removes that
    high-frequency jitter without noticeably changing the enclosed area or
    merging/splitting anything -- the kernel is small and local, well below
    the gap size between genuinely separate droplets.
    """
    ksize = max(3, int(ksize) | 1)
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return smoothed


def background_correct_and_segment(image_rgb: np.ndarray, params: SegmentationParams | None = None) -> SegmentationResult:
    params = params or SegmentationParams()
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    # The annealed droplets are consistently more saturated than the substrate,
    # even when the droplet core and rim have opposite brightness.
    denoised = cv2.medianBlur(saturation, 5)
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0)

    otsu_value, _ = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = int(np.clip(otsu_value + params.gray_threshold_offset, 0, 255))
    _, mask = cv2.threshold(denoised, threshold_value, 255, cv2.THRESH_BINARY)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_px = min(9, max(3, odd_kernel_size_from_um(3.0, image_rgb.shape)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = fill_holes(mask, max_hole_area_px=None)

    _, _, area_scale = pixel_calibration(image_rgb.shape)
    min_area_px = max(1, int(round(params.min_droplet_area_um2 / max(area_scale, 1e-9))))
    n_labels, raw_labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(mask)
    for label_id in range(1, n_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area_px:
            cleaned[raw_labels == label_id] = 255
    mask = cleaned
    mask = merge_small_edge_fragments(mask, params)

    # Smooth the jagged, noise-driven boundary into a physically plausible
    # curve. Run one more small-area cleanup afterwards in case smoothing
    # shrinks a thin sliver below the minimum droplet size.
    mask = smooth_mask_boundary(mask, ksize=odd_kernel_size_from_um(9.0, image_rgb.shape))
    n_labels, raw_labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(mask)
    for label_id in range(1, n_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area_px:
            cleaned[raw_labels == label_id] = 255
    mask = cleaned

    labels = connected_labels(mask)
    return SegmentationResult(image_rgb=image_rgb, gray_corrected=denoised, mask=mask, labels=labels)


def odd_kernel_size_from_um(size_um: float, image_shape: tuple[int, int, int] | tuple[int, int]) -> int:
    if size_um <= 0:
        return 1
    h, w = image_shape[:2]
    um_per_px = ((FIELD_WIDTH_UM / float(w)) + (FIELD_HEIGHT_UM / float(h))) / 2
    size_px = max(3, int(round(size_um / max(um_per_px, 1e-9))))
    if size_px % 2 == 0:
        size_px += 1
    return size_px


def fill_holes(mask: np.ndarray, max_hole_area_px: int | None = None) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    inv = 1 - binary
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 8)

    border_labels = set(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))

    filled = binary.copy()
    for label_id in range(1, n_labels):
        if label_id not in border_labels:
            if max_hole_area_px is not None and stats[label_id, cv2.CC_STAT_AREA] > max_hole_area_px:
                continue
            filled[labels == label_id] = 1
    return (filled * 255).astype(np.uint8)


def connected_labels(mask: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(labels, dtype=np.uint16)
    next_label = 1
    for label_id in range(1, n_labels):
        area_px = stats[label_id, cv2.CC_STAT_AREA]
        if area_px <= 0:
            continue
        cleaned[labels == label_id] = next_label
        next_label += 1
    return cleaned


def merge_small_edge_fragments(mask: np.ndarray, params: SegmentationParams | None = None) -> np.ndarray:
    """Fold tiny fragments back into the large neighbor they are actually
    part of.

    At sharp concavities/kinks on a fused droplet cluster's outline, the
    saturation ring can locally break into a hairline gap instead of staying
    fully closed. The sliver of ring pixels right next to the gap then forms
    its own tiny closed loop and gets counted as an independent droplet,
    even though it is geometrically just a piece of the big cluster's rim.

    We look for pairs where a small component sits within a few micrometers
    of a component many times its own size, and locally bridge just that gap
    (within the pair's own bounding box) so they become one connected region.
    This is deliberately local: it never touches the rest of the mask, so two
    separate droplets that are merely close to each other -- but comparable
    in size -- are left alone.
    """
    params = params or SegmentationParams()
    binary = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n_labels <= 2:
        return mask

    _, _, area_scale = pixel_calibration(mask.shape)
    min_area_px = max(1, int(round(params.min_droplet_area_um2 / max(area_scale, 1e-9))))
    small_area_cap_px = min_area_px * 6
    size_ratio_needed = 6.0
    bridge_px = max(5, odd_kernel_size_from_um(8.0, mask.shape))

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_px, bridge_px))
    dilated = cv2.dilate(binary, dilate_kernel, iterations=1)
    _, dilated_labels = cv2.connectedComponents(dilated, 8)[:2]

    groups: dict[int, list[int]] = {}
    for label_id in range(1, n_labels):
        ys, xs = np.where(labels == label_id)
        if ys.size == 0:
            continue
        d_id = int(dilated_labels[ys[0], xs[0]])
        groups.setdefault(d_id, []).append(label_id)

    merged = mask.copy()
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge_px, bridge_px))
    for member_ids in groups.values():
        if len(member_ids) < 2:
            continue
        member_ids_sorted = sorted(member_ids, key=lambda lid: stats[lid, cv2.CC_STAT_AREA], reverse=True)
        biggest = member_ids_sorted[0]
        biggest_area = stats[biggest, cv2.CC_STAT_AREA]
        fragments = [
            lid
            for lid in member_ids_sorted[1:]
            if stats[lid, cv2.CC_STAT_AREA] <= small_area_cap_px
            and biggest_area >= size_ratio_needed * stats[lid, cv2.CC_STAT_AREA]
        ]
        if not fragments:
            continue

        relevant_ids = [biggest] + fragments
        x0 = min(stats[lid, cv2.CC_STAT_LEFT] for lid in relevant_ids)
        y0 = min(stats[lid, cv2.CC_STAT_TOP] for lid in relevant_ids)
        x1 = max(stats[lid, cv2.CC_STAT_LEFT] + stats[lid, cv2.CC_STAT_WIDTH] for lid in relevant_ids)
        y1 = max(stats[lid, cv2.CC_STAT_TOP] + stats[lid, cv2.CC_STAT_HEIGHT] for lid in relevant_ids)
        pad = bridge_px
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(mask.shape[1], x1 + pad), min(mask.shape[0], y1 + pad)

        roi_component_mask = (np.isin(labels[y0:y1, x0:x1], relevant_ids).astype(np.uint8)) * 255
        bridged = cv2.morphologyEx(roi_component_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        merged[y0:y1, x0:x1] = np.maximum(merged[y0:y1, x0:x1], bridged)

    return merged


def pixel_calibration(image_shape: tuple[int, int, int]) -> tuple[float, float, float]:
    h, w = image_shape[:2]
    um_per_px_x = FIELD_WIDTH_UM / float(w)
    um_per_px_y = FIELD_HEIGHT_UM / float(h)
    area_um2_per_px = um_per_px_x * um_per_px_y
    return um_per_px_x, um_per_px_y, area_um2_per_px


def contour_for_component(component_mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def eccentricity_from_moments(component_mask: np.ndarray) -> float:
    moments = cv2.moments(component_mask.astype(np.uint8))
    if moments["m00"] == 0:
        return np.nan
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]
    common = math.sqrt(max((mu20 - mu02) ** 2 + 4 * mu11**2, 0.0))
    major = (mu20 + mu02 + common) / 2
    minor = (mu20 + mu02 - common) / 2
    if major <= 0 or minor < 0:
        return np.nan
    return math.sqrt(max(1.0 - minor / major, 0.0))


def extract_region_features(
    path: Path,
    group: str,
    seg: SegmentationResult,
    root: Path,
    params: SegmentationParams | None = None,
) -> pd.DataFrame:
    params = params or SegmentationParams()
    um_per_px_x, um_per_px_y, area_scale = pixel_calibration(seg.image_rgb.shape)
    h, w = seg.mask.shape

    labels = seg.labels
    ids = [i for i in np.unique(labels) if i != 0]
    rows = []

    centroids_px = {}
    for label_id in ids:
        ys, xs = np.where(labels == label_id)
        if len(xs) == 0:
            continue
        centroids_px[label_id] = (float(xs.mean()), float(ys.mean()))

    for label_id in ids:
        component = (labels == label_id).astype(np.uint8)
        area_px = int(component.sum())
        area_um2 = area_px * area_scale
        if area_um2 < params.min_droplet_area_um2:
            continue

        contour = contour_for_component(component)
        if contour is None:
            continue

        perimeter_px = float(cv2.arcLength(contour, True))
        perimeter_um = perimeter_px * (um_per_px_x + um_per_px_y) / 2
        circularity = 4 * math.pi * area_um2 / (perimeter_um**2) if perimeter_um > 0 else np.nan
        circularity = float(np.clip(circularity, 0, 1.2)) if np.isfinite(circularity) else np.nan

        contour_area_px = float(cv2.contourArea(contour))
        hull = cv2.convexHull(contour)
        hull_area_px = float(cv2.contourArea(hull))
        solidity = contour_area_px / hull_area_px if hull_area_px > 0 else np.nan
        solidity = float(np.clip(solidity, 0, 1)) if np.isfinite(solidity) else np.nan

        eccentricity = eccentricity_from_moments(component)
        x, y, bw, bh = cv2.boundingRect(contour)
        equivalent_diameter_um = 2 * math.sqrt(area_um2 / math.pi)
        aspect_ratio = bw / bh if bh > 0 else np.nan
        touches_border = x <= 1 or y <= 1 or (x + bw) >= (w - 1) or (y + bh) >= (h - 1)

        cx_px, cy_px = centroids_px[label_id]
        cx_um = cx_px * um_per_px_x
        cy_um = cy_px * um_per_px_y

        intensity_values = seg.gray_corrected[component > 0]
        mean_intensity = float(np.mean(intensity_values))
        std_intensity = float(np.std(intensity_values))

        rows.append(
            {
                "image": path.name,
                "image_path": str(path),
                "group": group,
                "label_id": int(label_id),
                "centroid_x_um": cx_um,
                "centroid_y_um": cy_um,
                "area_um2": area_um2,
                "equivalent_diameter_um": equivalent_diameter_um,
                "perimeter_um": perimeter_um,
                "circularity": circularity,
                "eccentricity": eccentricity,
                "solidity": solidity,
                "aspect_ratio": aspect_ratio,
                "bbox_width_um": bw * um_per_px_x,
                "bbox_height_um": bh * um_per_px_y,
                "mean_corrected_intensity": mean_intensity,
                "std_corrected_intensity": std_intensity,
                "touches_border": bool(touches_border),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    coords = df[["centroid_x_um", "centroid_y_um"]].to_numpy(float)
    df["nearest_neighbor_distance_um"] = nearest_neighbor_distances(coords)
    df["relative_path"] = df["image_path"].map(lambda s: str(Path(s).relative_to(root)))
    return df


def nearest_neighbor_distances(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return np.full(len(coords), np.nan)
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    distances[distances == 0] = np.inf
    return np.min(distances, axis=1)


def local_phi_features(mask: np.ndarray) -> tuple[float, float, list[float]]:
    values = []
    h, w = mask.shape
    for r in range(PHI_GRID_ROWS):
        y0 = int(round(r * h / PHI_GRID_ROWS))
        y1 = int(round((r + 1) * h / PHI_GRID_ROWS))
        for c in range(PHI_GRID_COLS):
            x0 = int(round(c * w / PHI_GRID_COLS))
            x1 = int(round((c + 1) * w / PHI_GRID_COLS))
            tile = mask[y0:y1, x0:x1]
            values.append(float(np.mean(tile > 0)))
    arr = np.asarray(values, dtype=float)
    cmax = float(np.max(arr)) if arr.size else np.nan
    cv_phi = float(np.std(arr) / np.mean(arr)) if arr.size and np.mean(arr) > 0 else np.nan
    return cmax, cv_phi, values


def image_summary(path: Path, group: str, seg: SegmentationResult, region_df: pd.DataFrame) -> dict:
    _, _, area_scale = pixel_calibration(seg.image_rgb.shape)
    field_area_um2 = FIELD_WIDTH_UM * FIELD_HEIGHT_UM
    droplet_area_um2 = float(np.sum(seg.mask > 0) * area_scale)
    phi = droplet_area_um2 / field_area_um2
    cmax, cv_phi, _ = local_phi_features(seg.mask)
    max_region_area_um2 = safe_max(region_df, "area_um2")
    max_region_area_fraction = max_region_area_um2 / field_area_um2 if np.isfinite(max_region_area_um2) else np.nan

    return {
        "image": path.name,
        "image_path": str(path),
        "group": group,
        "n_droplets": int(len(region_df)),
        "phi": phi,
        "Cmax": cmax,
        "CV_phi": cv_phi,
        "total_droplet_area_um2": droplet_area_um2,
        "mean_area_um2": safe_mean(region_df, "area_um2"),
        "median_area_um2": safe_median(region_df, "area_um2"),
        "mean_circularity": safe_mean(region_df, "circularity"),
        "mean_eccentricity": safe_mean(region_df, "eccentricity"),
        "mean_solidity": safe_mean(region_df, "solidity"),
        "mean_nearest_neighbor_distance_um": safe_mean(region_df, "nearest_neighbor_distance_um"),
        "max_region_area_um2": max_region_area_um2,
        "max_region_area_fraction": max_region_area_fraction,
    }


def filter_valid_images_by_quality(
    region_df: pd.DataFrame,
    image_df: pd.DataFrame,
    min_droplets: int = MIN_VALID_DROPLETS_PER_IMAGE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    image_df_all = image_df.copy()
    if image_df_all.empty or "n_droplets" not in image_df_all:
        return region_df.copy(), image_df_all, image_df_all.iloc[0:0].copy(), image_df_all

    reasons = []
    for _, row in image_df_all.iterrows():
        row_reasons = []
        n_droplets = int(row.get("n_droplets", 0) or 0)
        if n_droplets < int(min_droplets):
            row_reasons.append(f"droplets<{min_droplets}")
        reasons.append("; ".join(row_reasons))

    image_df_all["quality_reason"] = reasons
    valid_images = image_df_all["quality_reason"].eq("")
    image_df_all["used_for_analysis"] = valid_images
    image_df_all["quality_status"] = np.where(valid_images, "included", "excluded")

    excluded_image_df = image_df_all.loc[~valid_images].copy()
    filtered_image_df = image_df_all.loc[valid_images].copy()

    if region_df.empty or "image_path" not in region_df or filtered_image_df.empty:
        return region_df.iloc[0:0].copy(), filtered_image_df, excluded_image_df, image_df_all

    valid_paths = set(filtered_image_df["image_path"].astype(str))
    filtered_region_df = region_df.loc[region_df["image_path"].astype(str).isin(valid_paths)].copy()
    return filtered_region_df, filtered_image_df, excluded_image_df, image_df_all


def quality_filter_summary(
    all_image_df: pd.DataFrame,
    filtered_image_df: pd.DataFrame,
    excluded_image_df: pd.DataFrame,
    failed_count: int = 0,
) -> pd.DataFrame:
    total_success = int(len(all_image_df))
    kept_count = int(len(filtered_image_df))
    excluded_count = int(len(excluded_image_df))
    if kept_count + excluded_count != total_success:
        raise ValueError(
            "Quality-filter counts are inconsistent: "
            f"processed {total_success}, kept {kept_count}, excluded {excluded_count}."
        )

    rows = [
        {"item": "Successfully processed images", "count": total_success},
        {"item": "Images included in final analysis", "count": kept_count},
        {"item": "Images excluded by quality filter", "count": excluded_count},
        {"item": "Images that failed processing", "count": int(failed_count)},
        {"item": f"droplets<{MIN_VALID_DROPLETS_PER_IMAGE}", "count": count_reason_contains(excluded_image_df, "droplets<")},
    ]
    return pd.DataFrame(rows)


def count_reason_contains(df: pd.DataFrame, text: str) -> int:
    if df.empty or "quality_reason" not in df:
        return 0
    return int(df["quality_reason"].fillna("").astype(str).str.contains(text, regex=False).sum())


def safe_mean(df: pd.DataFrame, column: str) -> float:
    return float(df[column].mean()) if not df.empty and column in df else np.nan


def safe_median(df: pd.DataFrame, column: str) -> float:
    return float(df[column].median()) if not df.empty and column in df else np.nan


def safe_max(df: pd.DataFrame, column: str) -> float:
    return float(df[column].max()) if not df.empty and column in df else np.nan


def save_segmentation_overlay(seg: SegmentationResult, region_df: pd.DataFrame, output_path: Path) -> None:
    image = seg.image_rgb.copy()
    labels = seg.labels
    overlay = image.copy()
    rng = np.random.default_rng(RANDOM_SEED)

    for _, row in region_df.iterrows():
        label_id = int(row["label_id"])
        component = labels == label_id
        color = rng.integers(40, 255, size=3, dtype=np.uint8)
        overlay[component] = (0.45 * overlay[component] + 0.55 * color).astype(np.uint8)

    edges = cv2.Canny((seg.mask > 0).astype(np.uint8) * 255, 50, 150)
    overlay[edges > 0] = [255, 40, 40]

    for _, row in region_df.iterrows():
        x = int(row["centroid_x_um"] / FIELD_WIDTH_UM * image.shape[1])
        y = int(row["centroid_y_um"] / FIELD_HEIGHT_UM * image.shape[0])
        cv2.putText(overlay, str(int(row["label_id"])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    write_image_unicode(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def write_image_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def save_gray_and_binary_images(seg: SegmentationResult, gray_path: Path, binary_path: Path) -> None:
    gray = cv2.normalize(seg.gray_corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    binary = ((seg.mask > 0).astype(np.uint8) * 255)
    write_image_unicode(gray_path, gray)
    write_image_unicode(binary_path, binary)


def resize_for_preview(image_rgb: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale >= 1.0:
        return image_rgb.copy()
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)


def make_preview_panel(seg: SegmentationResult, view_mode: str) -> np.ndarray:
    gray_rgb = cv2.cvtColor(cv2.normalize(seg.gray_corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    binary_rgb = cv2.cvtColor(((seg.mask > 0).astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)

    overlay = seg.image_rgb.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    mask = seg.mask > 0
    overlay[mask] = (0.58 * overlay[mask] + 0.42 * green[mask]).astype(np.uint8)
    edges = cv2.Canny((seg.mask > 0).astype(np.uint8) * 255, 50, 150)
    overlay[edges > 0] = [255, 40, 40]

    if view_mode == "Corrected grayscale":
        return gray_rgb
    if view_mode == "Binary mask":
        return binary_rgb
    if view_mode == "Triptych":
        target_h = overlay.shape[0]
        gap = np.full((target_h, 8, 3), 230, dtype=np.uint8)
        return np.hstack([overlay, gap, gray_rgb, gap, binary_rgb])
    return overlay


def rgb_to_photo(image_rgb: np.ndarray, max_width: int = 1280, max_height: int = 760) -> ImageTk.PhotoImage:
    h, w = image_rgb.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(image_rgb))


def count_regions_for_status(seg: SegmentationResult, params: SegmentationParams) -> int:
    _, _, area_scale = pixel_calibration(seg.image_rgb.shape)
    min_area_px = max(1, int(round(params.min_droplet_area_um2 / area_scale)))
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats((seg.mask > 0).astype(np.uint8), 8)
    return sum(1 for label_id in range(1, n_labels) if stats[label_id, cv2.CC_STAT_AREA] >= min_area_px)


class SegmentationTuner:
    def __init__(self, image_path: Path | None = None, batch_folder: Path | None = None):
        self.root = Tk()
        self.root.title("Droplet Segmentation Parameter Tuner")
        self.root.geometry("1380x900")
        self.root.minsize(1100, 720)

        self.default_params = SegmentationParams()
        self.params = load_segmentation_params()
        self.image_path = image_path
        self.image_rgb: np.ndarray | None = None
        self.preview_rgb: np.ndarray | None = None
        self.pending_job = None
        self.photo = None
        self.last_panel_shape: tuple[int, int] | None = None
        self.numeric_entry_refs = []
        self.selected_group_folders: list[GroupFolder] = []
        if batch_folder is not None:
            self.selected_group_folders.append(GroupFolder(group=batch_folder.name, folder=batch_folder))
        self.start_batch_after_close = False
        self.window_closed = False
        self.result_params = self.params

        self.gray_offset = IntVar(value=self.params.gray_threshold_offset)
        self.view_mode = StringVar(value="Overlay")
        self.status_text = StringVar(value="Select an image to start tuning.")

        self.root.protocol("WM_DELETE_WINDOW", self.close_without_batch)
        self.build_layout()
        if self.image_path is None:
            self.choose_image()
        else:
            self.load_image(self.image_path)

    def build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        left_holder = ttk.Frame(main, width=370)
        left_holder.pack(side="left", fill="y", padx=(0, 10))
        left_holder.pack_propagate(False)
        left_buttons = ttk.Frame(left_holder)
        left_buttons.pack(side="right", fill="y")
        left_canvas = Canvas(left_holder, highlightthickness=0, width=335)
        left_scrollbar = Scrollbar(left_buttons, orient="vertical", command=left_canvas.yview, width=22)
        left = ttk.Frame(left_canvas)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        ttk.Button(left_buttons, text="▲", width=3, command=lambda: left_canvas.yview_scroll(-6, "units")).pack(fill="x")
        left_scrollbar.pack(fill="y", expand=True)
        ttk.Button(left_buttons, text="▼", width=3, command=lambda: left_canvas.yview_scroll(6, "units")).pack(fill="x")
        left_canvas.pack(side="left", fill="both", expand=True)

        def update_scroll_region(_event=None) -> None:
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
            left_canvas.itemconfigure(left_window, width=left_canvas.winfo_width())

        def on_mousewheel(event) -> None:
            left_canvas.yview_scroll(int(-3 * (event.delta / 120)), "units")

        left.bind("<Configure>", update_scroll_region)
        left_canvas.bind("<Configure>", update_scroll_region)
        left.bind("<MouseWheel>", on_mousewheel)
        left_canvas.bind("<MouseWheel>", on_mousewheel)
        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        ttk.Label(left, text="Parameter tuning", font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Button(left, text="Select image", command=self.choose_image).pack(fill="x", pady=3)
        ttk.Button(left, text="Segment single image and save outputs", command=self.save_single_image_outputs).pack(fill="x", pady=3)
        ttk.Button(left, text="Save current parameters", command=self.save_current_params).pack(fill="x", pady=3)
        ttk.Button(left, text="Load saved parameters", command=self.load_saved_params_from_file).pack(fill="x", pady=3)
        ttk.Button(left, text="Restore defaults", command=self.reset_all_defaults).pack(fill="x", pady=3)
        ttk.Button(left, text="Select group folders", command=self.choose_batch_folder).pack(fill="x", pady=3)
        ttk.Button(left, text="Confirm parameters and start batch processing", command=self.start_batch_analysis).pack(fill="x", pady=3)
        ttk.Button(left, text="Recompute", command=self.update_preview_now).pack(fill="x", pady=3)

        view_box = ttk.LabelFrame(left, text="Display mode", padding=8)
        view_box.pack(fill="x", pady=8)
        for text in ["Overlay", "Corrected grayscale", "Binary mask", "Triptych"]:
            ttk.Radiobutton(view_box, text=text, value=text, variable=self.view_mode, command=self.schedule_update).pack(anchor="w")

        self.add_slider(left, "S-channel threshold offset", self.gray_offset, -80, 80, 1, "Positive values are stricter; negative values expand droplet regions.", self.default_params.gray_threshold_offset)
        ttk.Label(left, textvariable=self.status_text, wraplength=310, foreground="#1f4e79").pack(anchor="w", pady=10)
        ttk.Label(left, text=f"Parameter file: {DEFAULT_PARAMS_FILE.name}", wraplength=310, foreground="#666666").pack(anchor="w")

        self.image_label = ttk.Label(right, anchor="center")
        self.image_label.pack(fill="both", expand=True)

    def add_slider(self, parent, label: str, variable, from_: float, to: float, step: float, hint: str, default_value: float) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=6)
        header = ttk.Frame(frame)
        header.pack(fill="x")
        ttk.Label(header, text=label).pack(side="left", anchor="w")
        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(2, 0))

        def format_value(value) -> str:
            if isinstance(variable, IntVar):
                return str(int(round(float(value))))
            return f"{float(value):.3g}"

        entry = Entry(controls, width=12, relief="solid", bd=1, takefocus=True, cursor="xterm")
        entry.insert(0, format_value(variable.get()))
        entry.pack(side="right")
        ttk.Button(controls, text="Apply", width=6, command=lambda: apply_entry_value()).pack(side="right", padx=(0, 6))
        ttk.Button(controls, text="Enter", width=6, command=lambda: prompt_entry_value()).pack(side="right", padx=(0, 6))
        ttk.Button(controls, text="Default", width=6, command=lambda: self.reset_variable(variable, default_value)).pack(side="right", padx=(0, 6))

        updating_entry = {"active": False}

        def refresh_label(*_):
            if updating_entry["active"]:
                return
            entry.delete(0, "end")
            entry.insert(0, format_value(variable.get()))

        def apply_entry_value(*_) -> None:
            text = entry.get().strip()
            try:
                value = float(text)
            except ValueError:
                self.status_text.set(f'Invalid value for "{label}". Enter a number and click Apply or press Enter.')
                return

            value = min(max(value, from_), to)
            if isinstance(variable, IntVar):
                value = int(round(value))
            else:
                if step > 0:
                    value = round(value / step) * step
                value = float(value)

            updating_entry["active"] = True
            variable.set(value)
            updating_entry["active"] = False
            entry.delete(0, "end")
            entry.insert(0, format_value(variable.get()))
            self.schedule_update()

        def prompt_entry_value() -> None:
            value = simpledialog.askfloat(
                title="Enter parameter",
                prompt=f"{label}\nRange: {from_} - {to}",
                initialvalue=float(variable.get()),
                minvalue=float(from_),
                maxvalue=float(to),
                parent=self.root,
            )
            if value is None:
                return
            entry.delete(0, "end")
            entry.insert(0, format_value(value))
            apply_entry_value()

        def focus_entry(event) -> None:
            entry.focus_force()
            entry.icursor(f"@{event.x}")

        variable.trace_add("write", refresh_label)
        refresh_label()
        entry.bind("<ButtonPress-1>", focus_entry)
        entry.bind("<Return>", apply_entry_value)
        entry.bind("<KP_Enter>", apply_entry_value)
        entry.bind("<FocusIn>", lambda _event: entry.selection_range(0, "end"))
        self.numeric_entry_refs.append((entry, variable, refresh_label, apply_entry_value))
        def on_scale_change(_value) -> None:
            refresh_label()
            self.schedule_update()

        scale = ttk.Scale(frame, from_=from_, to=to, variable=variable, command=on_scale_change)
        scale.pack(fill="x")
        ttk.Label(frame, text=hint, foreground="#666666", wraplength=300).pack(anchor="w")

    def reset_variable(self, variable, default_value) -> None:
        variable.set(default_value)
        self.schedule_update()

    def reset_all_defaults(self) -> None:
        self.apply_params_to_controls(self.default_params)
        self.status_text.set("All default parameters have been restored. Review the preview.")
        self.schedule_update()

    def apply_params_to_controls(self, params: SegmentationParams) -> None:
        self.gray_offset.set(params.gray_threshold_offset)
        self.result_params = params

    def load_saved_params_from_file(self) -> None:
        initial_dir = DEFAULT_PARAMS_FILE.parent
        initial_file = DEFAULT_PARAMS_FILE.name if DEFAULT_PARAMS_FILE.exists() else ""
        path = filedialog.askopenfilename(
            title="Select a previously saved parameter file",
            initialdir=str(initial_dir),
            initialfile=initial_file,
            filetypes=[("JSON parameter files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            params = load_segmentation_params(Path(path))
            self.apply_params_to_controls(params)
            self.status_text.set(f"Loaded parameter file: {path}\nReview the preview.")
            self.schedule_update()
        except Exception as exc:
            messagebox.showerror("Load failed", f"Could not load parameter file:\n{path}\n\n{exc}")

    def current_params(self) -> SegmentationParams:
        return SegmentationParams(
            gray_threshold_offset=int(self.gray_offset.get()),
        )

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select a microscope image for tuning",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"), ("All files", "*.*")],
        )
        if path:
            self.load_image(Path(path))

    def save_single_image_outputs(self) -> None:
        if self.image_path is None:
            self.choose_image()
            if self.image_path is None:
                return

        params = self.current_params()
        self.status_text.set("Running single-image recognition and saving outputs...")
        self.root.update_idletasks()
        try:
            output_dir = run_single_image_recognition(self.image_path, params=params)
            self.status_text.set(
                f"Single-image recognition complete.\n"
                f"Output folder: {output_dir}\n"
                "Saved: corrected grayscale image, binary mask, and segmentation overlay."
            )
            messagebox.showinfo(
                "Single-image recognition complete",
                f"Saved the corrected grayscale image, binary mask, and segmentation overlay:\n{output_dir}",
                parent=self.root,
            )
        except Exception as exc:
            self.status_text.set(f"Single-image recognition failed: {exc}")
            messagebox.showerror("Single-image recognition failed", str(exc), parent=self.root)

    def choose_batch_folder(self) -> None:
        groups: list[GroupFolder] = []
        while True:
            folder = filedialog.askdirectory(title="Select an experimental-group image folder")
            if not folder:
                break

            folder_path = Path(folder)
            default_group = folder_path.name
            group_name = simpledialog.askstring(
                title="Name group",
                prompt=f"Name this experimental group folder:\n{folder_path}\n\nExamples: Opz, Opz-2, SAM",
                initialvalue=default_group,
                parent=self.root,
            )
            if group_name is None:
                break
            group_name = group_name.strip()
            if not group_name:
                messagebox.showwarning("Empty group name", "Group name cannot be empty. Select and name the folder again.")
                continue
            if any(item.group.lower() == group_name.lower() for item in groups):
                messagebox.showwarning("Duplicate group name", f"A group named {group_name} already exists. Choose another name.")
                continue

            groups.append(GroupFolder(group=group_name, folder=folder_path))
            add_more = messagebox.askyesno("Add another group", "Select another experimental group folder?")
            if not add_more:
                break

        if groups:
            self.selected_group_folders = groups
            group_text = "\n".join(f"{item.group}: {item.folder}" for item in groups)
            self.status_text.set(
                f"Selected {len(groups)} experimental group(s):\n{group_text}\n"
                "Continue tuning the parameters. When satisfied, click \"Confirm parameters and start batch processing\"."
            )

    def start_batch_analysis(self) -> None:
        if not self.selected_group_folders:
            self.choose_batch_folder()
            if not self.selected_group_folders:
                return

        params = self.save_current_params(show_message=False)
        self.result_params = params
        self.start_batch_after_close = True
        self.close_window()

    def close_without_batch(self) -> None:
        self.start_batch_after_close = False
        self.close_window()

    def close_window(self) -> None:
        if self.window_closed:
            return
        self.window_closed = True
        if self.pending_job is not None:
            try:
                self.root.after_cancel(self.pending_job)
            except Exception:
                pass
            self.pending_job = None
        self.root.quit()

    def load_image(self, path: Path) -> None:
        try:
            self.image_path = path
            self.image_rgb = read_image(path)
            self.preview_rgb = resize_for_preview(self.image_rgb)
            self.root.title(f"Droplet Segmentation Parameter Tuner - {path.name}")
            self.schedule_update()
        except Exception as exc:
            messagebox.showerror("Read failed", f"Could not read image:\n{path}\n\n{exc}")

    def schedule_update(self) -> None:
        if self.pending_job is not None:
            self.root.after_cancel(self.pending_job)
        self.pending_job = self.root.after(250, self.update_preview_now)

    def update_preview_now(self) -> None:
        self.pending_job = None
        if self.preview_rgb is None:
            return
        params = self.current_params()
        self.status_text.set("Computing preview...")
        self.root.update_idletasks()
        try:
            seg = background_correct_and_segment(self.preview_rgb, params)
            panel = make_preview_panel(seg, self.view_mode.get())
            self.last_panel_shape = panel.shape[:2]
            self.photo = rgb_to_photo(panel)
            self.image_label.configure(image=self.photo)
            phi = float(np.mean(seg.mask > 0))
            n_regions = count_regions_for_status(seg, params)
            self.status_text.set(
                f"Current image: {self.image_path.name if self.image_path else ''}\n"
                f"Detected regions: {n_regions}    Area fraction phi: {phi:.4f}\n"
                f"S-channel threshold offset: {params.gray_threshold_offset}"
            )
        except Exception as exc:
            self.status_text.set(f"Preview calculation failed: {exc}")

    def save_current_params(self, show_message: bool = True) -> SegmentationParams:
        params = self.current_params()
        save_segmentation_params(params)
        self.result_params = params
        if show_message:
            messagebox.showinfo("Saved", f"Parameters saved to:\n{DEFAULT_PARAMS_FILE}")
        return params

    def run(self) -> None:
        try:
            self.root.mainloop()
        finally:
            self.cleanup_tk()

    def cleanup_tk(self) -> None:
        if self.pending_job is not None:
            try:
                self.root.after_cancel(self.pending_job)
            except Exception:
                pass
            self.pending_job = None
        self.photo = None
        self.numeric_entry_refs.clear()
        try:
            self.root.destroy()
        except Exception:
            pass


def run_parameter_tuner(
    image_path: Path | None = None,
    batch_folder: Path | None = None,
) -> tuple[bool, list[GroupFolder], SegmentationParams]:
    tuner = SegmentationTuner(image_path, batch_folder)
    tuner.run()
    return tuner.start_batch_after_close, tuner.selected_group_folders, tuner.result_params


def estimate_a0(region_df: pd.DataFrame) -> tuple[float, pd.DataFrame, str]:
    candidates = region_df.copy()
    if EXCLUDE_BORDER_FOR_A0_AND_Q and "touches_border" in candidates:
        candidates = candidates[~candidates["touches_border"]]
    candidates = candidates.replace([np.inf, -np.inf], np.nan).dropna(subset=["area_um2", "circularity", "solidity", "eccentricity"])

    if len(candidates) < 10:
        return np.nan, candidates, "A0 cannot be estimated: too few valid droplets."

    area_cut = candidates["area_um2"].quantile(0.40)
    circ_cut = max(0.80, candidates["circularity"].quantile(0.60))
    solid_cut = max(0.90, candidates["solidity"].quantile(0.40))

    standard = candidates[
        (candidates["area_um2"] <= area_cut)
        & (candidates["circularity"] >= circ_cut)
        & (candidates["solidity"] >= solid_cut)
        & (candidates["eccentricity"] <= 0.75)
    ].copy()

    if len(standard) < 8:
        standard = candidates[
            (candidates["area_um2"] <= candidates["area_um2"].quantile(0.50))
            & (candidates["circularity"] >= candidates["circularity"].quantile(0.50))
        ].copy()

    if len(standard) < 5:
        return np.nan, standard, "A0 cannot be estimated: too few small, round standard droplets."

    a0 = float(standard["area_um2"].median())
    msg = f"A0={a0:.3f} um^2, estimated as the median area of {len(standard)} small, round regions."
    return a0, standard, msg


def gaussian_pdf(x: np.ndarray, mean: float, var: float, weight: float) -> np.ndarray:
    sd = math.sqrt(max(var, 1e-12))
    return weight * np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def component_intersection(gmm: GaussianMixture) -> float | None:
    means = gmm.means_.ravel()
    vars_ = gmm.covariances_.reshape(-1)
    weights = gmm.weights_.ravel()
    order = np.argsort(means)
    m1, m2 = means[order]
    v1, v2 = vars_[order]
    w1, w2 = weights[order]

    xs = np.linspace(m1, m2, 2000)
    diff = gaussian_pdf(xs, m1, v1, w1) - gaussian_pdf(xs, m2, v2, w2)
    sign_change = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]
    if len(sign_change) == 0:
        return None
    roots = xs[sign_change]
    return float(roots[np.argmin(np.abs(roots - np.mean([m1, m2])))] )


def has_clear_histogram_valley(values: np.ndarray, q: float) -> bool:
    if len(values) < 80 or not np.isfinite(q):
        return False
    lo, hi = np.percentile(values, [1, 99])
    if not (lo < q < hi):
        return False

    counts, edges = np.histogram(values[(values >= lo) & (values <= hi)], bins=70)
    centers = (edges[:-1] + edges[1:]) / 2
    smooth = np.convolve(counts.astype(float), np.array([1, 2, 3, 2, 1], dtype=float) / 9.0, mode="same")
    q_idx = int(np.argmin(np.abs(centers - q)))
    if q_idx < 3 or q_idx > len(smooth) - 4:
        return False

    left = smooth[:q_idx]
    right = smooth[q_idx + 1 :]
    if left.size < 3 or right.size < 3:
        return False

    left_peak = float(np.max(left))
    right_peak = float(np.max(right))
    valley = float(np.mean(smooth[max(0, q_idx - 1) : min(len(smooth), q_idx + 2)]))
    if left_peak <= 0 or right_peak <= 0:
        return False

    left_peak_idx = int(np.argmax(left))
    right_peak_idx = q_idx + 1 + int(np.argmax(right))
    has_left_peak = 0 < left_peak_idx < q_idx - 1 and left_peak > max(smooth[left_peak_idx - 1], smooth[left_peak_idx + 1])
    has_right_peak = q_idx + 2 < right_peak_idx < len(smooth) - 1 and right_peak > max(smooth[right_peak_idx - 1], smooth[right_peak_idx + 1])
    if not (has_left_peak and has_right_peak):
        return False

    return valley < min(left_peak, right_peak) * 0.80


def q_reliability(values: np.ndarray, gmm1: GaussianMixture, gmm2: GaussianMixture, q: float | None) -> tuple[bool, str]:
    if q is None:
        return False, "q_ML cannot be reliably identified: the two Gaussian components do not have a stable intersection."

    bic1 = gmm1.bic(values.reshape(-1, 1))
    bic2 = gmm2.bic(values.reshape(-1, 1))
    bic_gain = bic1 - bic2
    means = np.sort(gmm2.means_.ravel())
    vars_sorted = gmm2.covariances_.reshape(-1)[np.argsort(gmm2.means_.ravel())]
    weights_sorted = gmm2.weights_.ravel()[np.argsort(gmm2.means_.ravel())]
    pooled_sd = math.sqrt(np.mean(vars_sorted))
    separation = abs(means[1] - means[0]) / max(pooled_sd, 1e-9)
    min_weight = float(np.min(weights_sorted))
    mean_gap = abs(means[1] - means[0])
    max_sd = math.sqrt(float(np.max(vars_sorted)))
    wide_tail_ratio = max_sd / max(mean_gap, 1e-9)

    if bic_gain < 10:
        return False, f"q_ML cannot be reliably identified: the BIC gain for the bimodal model is too small, BIC gain={bic_gain:.2f}."
    if separation < 1.25:
        return False, f"q_ML cannot be reliably identified: the two ln(Ai/A0) distributions are insufficiently separated, separation={separation:.2f}."
    if min_weight < 0.08:
        return False, f"q_ML cannot be reliably identified: the smaller component weight is too low, weight={min_weight:.2f}."
    if wide_tail_ratio > 0.85:
        return False, f"q_ML cannot be reliably identified: the larger component is too broad, suggesting a unimodal long tail rather than a clear bimodal distribution, wide_tail_ratio={wide_tail_ratio:.2f}."
    if not has_clear_histogram_valley(values, q):
        return False, "q_ML cannot be reliably identified: the ln(Ai/A0) histogram does not show a clear bimodal valley."
    if not (np.percentile(values, 5) <= q <= np.percentile(values, 95)):
        return False, f"q_ML cannot be reliably identified: the intersection {q:.3f} lies outside the main data range."

    return True, f"q_ML reliably identified: BIC gain={bic_gain:.2f}, separation={separation:.2f}, min_weight={min_weight:.2f}."


def log_area_values_for_gmm(region_df: pd.DataFrame, a0: float) -> np.ndarray:
    if region_df.empty or not np.isfinite(a0) or a0 <= 0:
        return np.array([], dtype=float)

    df = region_df.copy()
    area = df["area_um2"].to_numpy(float)
    valid_mask = np.isfinite(area) & (area > 0)
    if EXCLUDE_BORDER_FOR_A0_AND_Q and "touches_border" in df:
        valid_mask &= ~df["touches_border"].to_numpy(bool)

    area_multiple = (df.loc[valid_mask, "area_um2"] / a0).replace([np.inf, -np.inf], np.nan).dropna()
    area_multiple = area_multiple[area_multiple > 0.2]
    if len(area_multiple) >= 20:
        area_multiple = area_multiple[area_multiple < area_multiple.quantile(0.995)]
    return np.log(area_multiple.to_numpy(float))


def fit_gmm_q_log(x_log: np.ndarray, random_state: int = RANDOM_SEED) -> tuple[float | None, GaussianMixture | None, GaussianMixture | None]:
    x_log = np.asarray(x_log, dtype=float)
    x_log = x_log[np.isfinite(x_log)]
    if len(x_log) < 30:
        return None, None, None

    x = x_log.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=random_state).fit(x)
    gmm2 = GaussianMixture(n_components=2, random_state=random_state, n_init=10).fit(x)
    q_log = component_intersection(gmm2)
    return q_log, gmm1, gmm2


def bootstrap_q_by_image(region_df: pd.DataFrame, a0: float) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    if region_df.empty or "image_path" not in region_df:
        return np.array([], dtype=float)

    image_paths = region_df["image_path"].dropna().astype(str).unique()
    if len(image_paths) < 2:
        return np.array([], dtype=float)

    grouped = {path: part for path, part in region_df.groupby(region_df["image_path"].astype(str))}
    qs = []
    for _ in range(N_BOOTSTRAP):
        sampled_paths = rng.choice(image_paths, size=len(image_paths), replace=True)
        sample_df = pd.concat([grouped[path] for path in sampled_paths], ignore_index=True)
        x_log = log_area_values_for_gmm(sample_df, a0)
        try:
            q_log, _, _ = fit_gmm_q_log(x_log, random_state=RANDOM_SEED)
            if q_log is not None and np.isfinite(q_log):
                qs.append(q_log)
        except Exception:
            continue
    return np.asarray(qs, dtype=float)


def fit_unified_ml_model(region_df: pd.DataFrame, image_df: pd.DataFrame, a0: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    df = region_df.copy()
    image_results = image_df.copy()
    df["area_multiple_Ai_A0"] = df["area_um2"] / a0 if np.isfinite(a0) and a0 > 0 else np.nan
    df["ln_Ai_A0"] = np.log(df["area_multiple_Ai_A0"].where(df["area_multiple_Ai_A0"] > 0))
    df["p_coal_ML"] = np.nan
    df["gmm_area_class_ML"] = "unclassified"

    x_log = log_area_values_for_gmm(df, a0)
    q_log, gmm1, gmm2 = fit_gmm_q_log(x_log)
    gmm_available = gmm2 is not None
    q_available = q_log is not None and np.isfinite(q_log) and gmm_available

    reason = "Too few valid ln(Ai/A0) values for stable two-component GMM fitting."
    status = "q_ML_unavailable"
    if gmm_available and gmm1 is not None:
        reliable, reason = q_reliability(x_log, gmm1, gmm2, q_log)
        if q_available and reliable:
            status = "reliable_gmm_intersection"
        elif q_available:
            status = "gmm_intersection"
        else:
            status = "gmm_without_stable_intersection"
        finite_mask = df["ln_Ai_A0"].replace([np.inf, -np.inf], np.nan).notna()
        x_all = df.loc[finite_mask, ["ln_Ai_A0"]].to_numpy(float)
        probs = gmm2.predict_proba(x_all)
        labels = gmm2.predict(x_all)
        high_component = int(np.argmax(gmm2.means_.ravel()))
        df.loc[finite_mask, "p_coal_ML"] = probs[:, high_component]
        df.loc[finite_mask, "gmm_area_class_ML"] = np.where(labels == high_component, "large_coalesced", "small_discrete")

    q_boot = bootstrap_q_by_image(df, a0)
    if len(q_boot) >= max(50, N_BOOTSTRAP * 0.30):
        q_ci_low, q_ci_high = np.percentile(q_boot, [2.5, 97.5])
    else:
        q_ci_low, q_ci_high = np.nan, np.nan
        if q_available:
            reason += " Image-level bootstrap produced too few valid q_ML estimates; the 95% confidence interval is not reported."

    image_factors = []
    for image_path, part in df.groupby("image_path"):
        area = part["area_um2"].replace([np.inf, -np.inf], np.nan)
        p = part["p_coal_ML"].replace([np.inf, -np.inf], np.nan)
        valid = area.notna() & p.notna() & (area > 0)
        area_sum = float(area[valid].sum())
        f_coal = float((p[valid] * area[valid]).sum() / area_sum) if area_sum > 0 else np.nan
        image_factors.append({"image_path": image_path, "F_coal_ML": f_coal})

    factor_df = pd.DataFrame(image_factors)
    if not factor_df.empty:
        image_results = image_results.merge(factor_df, on="image_path", how="left")
    else:
        image_results["F_coal_ML"] = np.nan

    image_results["Phi_U"] = 1.0 / (1.0 + image_results["CV_phi"])
    film_product = image_results["phi"] * image_results["F_coal_ML"] * image_results["Phi_U"]
    image_results["Phi_ML"] = np.where(film_product >= 0, np.cbrt(film_product), np.nan)
    image_results["q_global_log"] = float(q_log) if q_available else np.nan
    image_results["q_global_area_multiple"] = float(np.exp(q_log)) if q_available else np.nan

    means = np.sort(gmm2.means_.ravel()) if gmm2 is not None else np.array([np.nan, np.nan])
    order = np.argsort(gmm2.means_.ravel()) if gmm2 is not None else np.array([0, 1])
    variances = gmm2.covariances_.reshape(-1)[order] if gmm2 is not None else np.array([np.nan, np.nan])
    weights = gmm2.weights_.ravel()[order] if gmm2 is not None else np.array([np.nan, np.nan])
    model_parameters = pd.DataFrame(
        [
            {
                "model_version": "droplet_ml_pipeline_release",
                "A0_um2": float(a0) if np.isfinite(a0) else np.nan,
                "q_ML_log": float(q_log) if q_available else np.nan,
                "q_ML_area_multiple": float(np.exp(q_log)) if q_available else np.nan,
                "q_global": float(np.exp(q_log)) if q_available else np.nan,
                "q_ci95_log_low": float(q_ci_low) if np.isfinite(q_ci_low) else np.nan,
                "q_ci95_log_high": float(q_ci_high) if np.isfinite(q_ci_high) else np.nan,
                "q_ci95_area_multiple_low": float(np.exp(q_ci_low)) if np.isfinite(q_ci_low) else np.nan,
                "q_ci95_area_multiple_high": float(np.exp(q_ci_high)) if np.isfinite(q_ci_high) else np.nan,
                "q_status": status,
                "q_reason": reason,
                "gmm_small_mean_log": float(means[0]),
                "gmm_large_mean_log": float(means[1]),
                "gmm_small_variance_log": float(variances[0]),
                "gmm_large_variance_log": float(variances[1]),
                "gmm_small_weight": float(weights[0]),
                "gmm_large_weight": float(weights[1]),
                "n_regions_for_gmm": int(len(x_log)),
                "n_images_for_bootstrap": int(df["image_path"].nunique()) if "image_path" in df else 0,
                "bootstrap_success_count": int(len(q_boot)),
                "final_parameters_for_dynamics": "q_global, phi, F_coal_ML, CV_phi",
            }
        ]
    )
    return df, image_results, model_parameters, q_boot


def build_sample_summary(image_results: pd.DataFrame, model_parameters: pd.DataFrame) -> pd.DataFrame:
    if image_results.empty or "group" not in image_results:
        return pd.DataFrame()

    metrics = ["phi", "F_coal_ML", "CV_phi", "Phi_U", "Phi_ML", "Cmax"]
    rows = []
    model_row = model_parameters.iloc[0].to_dict() if not model_parameters.empty else {}
    for group, part in image_results.groupby("group"):
        row = {
            "sample": group,
            "group": group,
            "n_images": int(len(part)),
            "q_global": model_row.get("q_global", np.nan),
            "q_ML_log": model_row.get("q_ML_log", np.nan),
            "A0_um2": model_row.get("A0_um2", np.nan),
        }
        for metric in metrics:
            values = part[metric].replace([np.inf, -np.inf], np.nan).dropna() if metric in part else pd.Series(dtype=float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_dynamics_input_table(image_results: pd.DataFrame, model_parameters: pd.DataFrame) -> pd.DataFrame:
    if image_results.empty or "group" not in image_results:
        return pd.DataFrame()

    model_row = model_parameters.iloc[0].to_dict() if not model_parameters.empty else {}
    rows = []
    for group, part in image_results.groupby("group"):
        rows.append(
            {
                "sample": group,
                "q_global": model_row.get("q_global", np.nan),
                "phi": float(part["phi"].replace([np.inf, -np.inf], np.nan).mean()),
                "F_coal_ML": float(part["F_coal_ML"].replace([np.inf, -np.inf], np.nan).mean()),
                "CV_phi": float(part["CV_phi"].replace([np.inf, -np.inf], np.nan).mean()),
            }
        )
    return pd.DataFrame(rows, columns=["sample", "q_global", "phi", "F_coal_ML", "CV_phi"])


def make_pca_plot(region_df: pd.DataFrame, output_path: Path) -> None:
    features = [
        "area_um2",
        "equivalent_diameter_um",
        "circularity",
        "eccentricity",
        "solidity",
        "aspect_ratio",
        "nearest_neighbor_distance_um",
        "mean_corrected_intensity",
        "std_corrected_intensity",
    ]
    df = region_df.dropna(subset=features + ["group"]).copy()
    if len(df) < 5:
        return

    x = StandardScaler().fit_transform(df[features])
    pcs = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(x)
    plot_df = pd.DataFrame(
        {
            "PC1": pcs[:, 0],
            "PC2": pcs[:, 1],
            "group": df["group"].values,
            "gmm_area_class_ML": df.get("gmm_area_class_ML", pd.Series("unclassified", index=df.index)).values,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="group", style="gmm_area_class_ML", s=35, alpha=0.8)
    plt.title("PCA of droplet features")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_q_distribution(region_df: pd.DataFrame, model_parameters: pd.DataFrame, q_boot: np.ndarray, output_path: Path) -> None:
    if "area_multiple_Ai_A0" not in region_df:
        return

    values = region_df["area_multiple_Ai_A0"].replace([np.inf, -np.inf], np.nan).dropna()
    values = values[(values > 0) & (values < values.quantile(0.995))]
    if len(values) < 5:
        return

    params = model_parameters.iloc[0].to_dict() if not model_parameters.empty else {}
    q_area = params.get("q_ML_area_multiple", np.nan)
    q_log = params.get("q_ML_log", np.nan)
    q_ci_low = params.get("q_ci95_log_low", np.nan)
    q_ci_high = params.get("q_ci95_log_high", np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_df = region_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["area_multiple_Ai_A0", "group"]).copy()
    plot_df = plot_df[(plot_df["area_multiple_Ai_A0"] > 0) & (plot_df["area_multiple_Ai_A0"] < plot_df["area_multiple_Ai_A0"].quantile(0.995))]
    sns.histplot(data=plot_df, x="area_multiple_Ai_A0", hue="group", bins=45, kde=True, ax=axes[0], element="step", common_norm=False)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Area multiple Ai/A0")
    axes[0].set_title("Opz area multiple distribution")
    if np.isfinite(q_area):
        axes[0].axvline(q_area, color="#CC3311", lw=2, label=f"q_ML={q_area:.3f}")
        axes[0].legend()

    if len(q_boot) > 0:
        sns.histplot(q_boot, bins=35, kde=True, ax=axes[1], color="#44AA99")
        if np.isfinite(q_ci_low):
            axes[1].axvspan(q_ci_low, q_ci_high, color="#DDCC77", alpha=0.35, label="95% CI")
            axes[1].legend()
    if np.isfinite(q_log):
        axes[1].axvline(q_log, color="#CC3311", lw=2)
    axes[1].set_xlabel("Bootstrap q_ML = ln(Ai/A0)")
    axes[1].set_title("Image-level bootstrap q_ML")

    fig.suptitle(params.get("q_status", "q_ML result"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def gmm_plot_data(region_df: pd.DataFrame, model_parameters: pd.DataFrame) -> pd.DataFrame:
    if region_df.empty or model_parameters.empty or "ln_Ai_A0" not in region_df:
        return pd.DataFrame()

    params = model_parameters.iloc[0].to_dict()
    required = [
        "gmm_small_mean_log",
        "gmm_large_mean_log",
        "gmm_small_variance_log",
        "gmm_large_variance_log",
        "gmm_small_weight",
        "gmm_large_weight",
    ]
    if any(not np.isfinite(params.get(key, np.nan)) for key in required):
        return pd.DataFrame()

    x_values = region_df["ln_Ai_A0"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(x_values) < 5:
        return pd.DataFrame()

    lo, hi = np.percentile(x_values, [0.5, 99.5])
    pad = max((hi - lo) * 0.15, 0.5)
    x_grid = np.linspace(lo - pad, hi + pad, 500)
    small_density = gaussian_pdf(
        x_grid,
        params["gmm_small_mean_log"],
        params["gmm_small_variance_log"],
        params["gmm_small_weight"],
    )
    large_density = gaussian_pdf(
        x_grid,
        params["gmm_large_mean_log"],
        params["gmm_large_variance_log"],
        params["gmm_large_weight"],
    )
    total_density = small_density + large_density
    p_coal = np.divide(large_density, total_density, out=np.full_like(total_density, np.nan), where=total_density > 0)

    return pd.DataFrame(
        {
            "x_ln_Ai_A0": x_grid,
            "small_discrete_density": small_density,
            "large_coalesced_density": large_density,
            "gmm_total_density": total_density,
            "p_coal_ML": p_coal,
            "q_global_log": params.get("q_ML_log", np.nan),
            "q_global": params.get("q_global", np.nan),
        }
    )


def actual_distribution_data(region_df: pd.DataFrame) -> pd.DataFrame:
    if region_df.empty or "ln_Ai_A0" not in region_df:
        return pd.DataFrame()

    rows = []
    for group, part in region_df.groupby("group"):
        values = part["ln_Ai_A0"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if len(values) < 2:
            continue
        counts, edges = np.histogram(values, bins=45, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        for center, density in zip(centers, counts):
            rows.append({"group": group, "x_ln_Ai_A0": center, "probability_density": density})
    return pd.DataFrame(rows)


def area_distribution_histogram_data(region_df: pd.DataFrame, bins: int = 36) -> pd.DataFrame:
    columns = [
        "group",
        "bin_left_area_um2",
        "bin_right_area_um2",
        "A_i_um2",
        "bin_center_area_um2",
        "droplet_count",
        "probability_density",
    ]
    if region_df.empty or "area_um2" not in region_df:
        return pd.DataFrame(columns=columns)

    values_all = region_df["area_um2"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    values_all = values_all[values_all > 0]
    if len(values_all) < 2:
        return pd.DataFrame(columns=columns)

    lo, hi = np.percentile(values_all, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values_all)), float(np.max(values_all))
    lo = max(lo, float(np.min(values_all[values_all > 0])))
    edges = np.geomspace(lo, hi, bins + 1)

    rows = []
    for group, part in region_df.groupby("group"):
        values = part["area_um2"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        values = values[(values >= lo) & (values <= hi)]
        if len(values) == 0:
            continue
        counts, _ = np.histogram(values, bins=edges)
        density, _ = np.histogram(values, bins=edges, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        for left, right, center, count, dens in zip(edges[:-1], edges[1:], centers, counts, density):
            rows.append(
                {
                    "group": group,
                    "bin_left_area_um2": left,
                    "bin_right_area_um2": right,
                    "A_i_um2": center,
                    "bin_center_area_um2": center,
                    "droplet_count": int(count),
                    "probability_density": float(dens) if np.isfinite(dens) else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def plot_area_distribution_histogram(region_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    hist_df = area_distribution_histogram_data(region_df)
    if hist_df.empty:
        return hist_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.6))
    groups = list(hist_df["group"].dropna().astype(str).unique())
    palette = sns.color_palette("Set2", n_colors=max(1, len(groups)))
    group_colors = {group: palette[idx] for idx, group in enumerate(groups)}
    width = hist_df["bin_right_area_um2"] - hist_df["bin_left_area_um2"]
    offsets = np.linspace(-0.18, 0.18, len(groups)) if len(groups) > 1 else [0.0]

    for offset, group in zip(offsets, groups):
        part = hist_df[hist_df["group"].astype(str) == group]
        group_width = (part["bin_right_area_um2"] - part["bin_left_area_um2"]) * (0.72 / max(1, len(groups)))
        plt.bar(
            part["A_i_um2"] * (1 + offset),
            part["droplet_count"],
            width=group_width,
            color=group_colors[group],
            alpha=0.82,
            edgecolor="#555555",
            linewidth=0.35,
            label=group,
        )

    plt.xscale("log")
    plt.xlabel(r"Droplet area $A_i$ ($\mu m^2$)")
    plt.ylabel("Droplet count")
    plt.title("Droplet area distribution")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return hist_df


AREA_BIN_EDGES = [-np.inf, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, np.inf]
AREA_BIN_LABELS = [
    "x < -2.5",
    "-2.5 ~ -2.0",
    "-2.0 ~ -1.5",
    "-1.5 ~ -1.0",
    "-1.0 ~ -0.5",
    "-0.5 ~ 0.0",
    "0.0 ~ 0.5",
    "0.5 ~ 1.0",
    "1.0 ~ 1.5",
    "1.5 ~ 2.0",
    "x > 2.0",
]


def droplet_state_composition_data(region_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "group",
        "bin_index",
        "area_bin",
        "bin_left_ln_Ai_A0",
        "bin_right_ln_Ai_A0",
        "mean_fraction_per_image",
        "mean_p_coal_ML",
        "total_droplets_in_bin",
        "n_images",
    ]
    required = ["group", "image_path", "ln_Ai_A0", "p_coal_ML"]
    if region_df.empty or any(column not in region_df for column in required):
        return pd.DataFrame(columns=columns)

    df = region_df[required].replace([np.inf, -np.inf], np.nan).dropna(subset=["group", "image_path", "ln_Ai_A0"]).copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["area_bin"] = pd.cut(df["ln_Ai_A0"], bins=AREA_BIN_EDGES, labels=AREA_BIN_LABELS, include_lowest=True, right=False)
    rows = []
    for group, group_df in df.groupby("group"):
        image_paths = sorted(group_df["image_path"].astype(str).unique())
        n_images = len(image_paths)
        for bin_index, bin_label in enumerate(AREA_BIN_LABELS):
            per_image_fractions = []
            per_image_probabilities = []
            total_bin_count = 0
            for image_path in image_paths:
                image_df = group_df[group_df["image_path"].astype(str) == image_path]
                n_total = len(image_df)
                if n_total == 0:
                    continue
                bin_df = image_df[image_df["area_bin"].astype(str) == bin_label]
                n_bin = len(bin_df)
                total_bin_count += n_bin
                per_image_fractions.append(n_bin / n_total)
                p_values = bin_df["p_coal_ML"].replace([np.inf, -np.inf], np.nan).dropna()
                if len(p_values):
                    per_image_probabilities.append(float(p_values.mean()))

            rows.append(
                {
                    "group": group,
                    "bin_index": bin_index,
                    "area_bin": bin_label,
                    "bin_left_ln_Ai_A0": AREA_BIN_EDGES[bin_index],
                    "bin_right_ln_Ai_A0": AREA_BIN_EDGES[bin_index + 1],
                    "mean_fraction_per_image": float(np.mean(per_image_fractions)) if per_image_fractions else np.nan,
                    "mean_p_coal_ML": float(np.mean(per_image_probabilities)) if per_image_probabilities else np.nan,
                    "total_droplets_in_bin": int(total_bin_count),
                    "n_images": int(n_images),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def plot_droplet_state_composition_matrix(region_df: pd.DataFrame, model_parameters: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    plot_df = droplet_state_composition_data(region_df)
    if plot_df.empty:
        return plot_df

    groups = list(plot_df["group"].dropna().astype(str).unique())
    q_log = np.nan
    if not model_parameters.empty:
        q_log = model_parameters.iloc[0].to_dict().get("q_ML_log", np.nan)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, max(3.8, 1.0 + 0.85 * len(groups))))
    y_positions = {group: idx for idx, group in enumerate(groups)}
    x_positions = np.arange(len(AREA_BIN_LABELS))

    size_scale = 3200.0
    sizes = 18.0 + size_scale * plot_df["mean_fraction_per_image"].fillna(0).to_numpy(float)
    colors = plot_df["mean_p_coal_ML"].fillna(0).to_numpy(float)
    scatter = ax.scatter(
        plot_df["bin_index"],
        plot_df["group"].map(y_positions),
        s=sizes,
        c=colors,
        cmap="plasma",
        vmin=0,
        vmax=1,
        alpha=0.82,
        edgecolors="#626262",
        linewidths=0.35,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(AREA_BIN_LABELS, rotation=30, ha="right")
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(groups)
    ax.set_xlabel(r"Normalized droplet area  $x=\ln(A_i/A_0)$")
    ax.set_ylabel("Sample")
    ax.set_title("Droplet state composition across all images", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.grid(axis="y", linestyle="-", alpha=0.18)

    if np.isfinite(q_log):
        finite_edges = np.asarray(AREA_BIN_EDGES[1:-1], dtype=float)
        boundary_index = int(np.searchsorted(finite_edges, q_log, side="right"))
        ax.axvline(boundary_index - 0.5, color="#7B2CBF", linestyle="--", linewidth=1.5)
        ax.text(boundary_index - 0.5, len(groups) - 0.35, r"$x_q$", color="#7B2CBF", ha="center", va="bottom")

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r"Mean large-component posterior probability $\bar{p}_k$")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    legend_fractions = [0.001, 0.01, 0.05, 0.10, 0.20]
    handles = [
        ax.scatter([], [], s=18.0 + size_scale * fraction, color="#BDBDBD", edgecolors="#626262", linewidths=0.35)
        for fraction in legend_fractions
    ]
    labels = ["0.1%", "1%", "5%", "10%", "20%"]
    ax.legend(handles, labels, title=r"Bubble size $\bar{f}_k$", loc="upper left", bbox_to_anchor=(0.0, 1.26), ncol=len(labels), frameon=False)

    fig.subplots_adjust(left=0.12, right=0.86, top=0.72, bottom=0.25)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return plot_df


def droplet_plot_points(region_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["group", "image", "image_path", "area_um2", "area_multiple_Ai_A0", "ln_Ai_A0", "p_coal_ML", "gmm_area_class_ML"]
    if region_df.empty:
        return pd.DataFrame(columns=columns)
    available = [column for column in columns if column in region_df]
    return region_df[available].replace([np.inf, -np.inf], np.nan).dropna(subset=["ln_Ai_A0"], how="any").copy()


def droplet_state_map_data(region_df: pd.DataFrame, grid_cols: int = 12, grid_rows: int = 8) -> pd.DataFrame:
    columns = [
        "group",
        "grid_col",
        "grid_row",
        "center_x_um",
        "center_y_um",
        "mean_droplet_fraction_per_image",
        "mean_droplet_count_per_image",
        "p_coal_ML",
        "total_droplets_in_cell",
        "n_images",
    ]
    required = ["group", "image_path", "centroid_x_um", "centroid_y_um", "area_multiple_Ai_A0", "p_coal_ML"]
    if region_df.empty or any(column not in region_df for column in required):
        return pd.DataFrame(columns=columns)

    records = []
    for group, group_df in region_df.groupby("group"):
        group_df = group_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["image_path", "centroid_x_um", "centroid_y_um", "p_coal_ML"]
        ).copy()
        image_paths = sorted(group_df["image_path"].astype(str).unique())
        n_images = len(image_paths)
        if n_images == 0:
            continue

        group_df["grid_col"] = np.clip((group_df["centroid_x_um"] / FIELD_WIDTH_UM * grid_cols).astype(int), 0, grid_cols - 1)
        group_df["grid_row"] = np.clip((group_df["centroid_y_um"] / FIELD_HEIGHT_UM * grid_rows).astype(int), 0, grid_rows - 1)

        image_totals = group_df.groupby(group_df["image_path"].astype(str)).size().to_dict()
        for row_idx in range(grid_rows):
            for col_idx in range(grid_cols):
                cell_df = group_df[(group_df["grid_row"] == row_idx) & (group_df["grid_col"] == col_idx)]
                per_image_fractions = []
                per_image_counts = []
                for image_path in image_paths:
                    image_df = group_df[group_df["image_path"].astype(str) == image_path]
                    total = image_totals.get(image_path, 0)
                    count = int(((image_df["grid_row"] == row_idx) & (image_df["grid_col"] == col_idx)).sum())
                    if total > 0:
                        per_image_fractions.append(count / total)
                    per_image_counts.append(count)
                p_values = cell_df["p_coal_ML"].dropna()
                records.append(
                    {
                        "group": group,
                        "grid_col": col_idx,
                        "grid_row": row_idx,
                        "center_x_um": (col_idx + 0.5) * FIELD_WIDTH_UM / grid_cols,
                        "center_y_um": (row_idx + 0.5) * FIELD_HEIGHT_UM / grid_rows,
                        "mean_droplet_fraction_per_image": float(np.mean(per_image_fractions)) if per_image_fractions else np.nan,
                        "mean_droplet_count_per_image": float(np.mean(per_image_counts)) if per_image_counts else np.nan,
                        "p_coal_ML": float(np.clip(p_values.mean(), 0, 1)) if len(p_values) else np.nan,
                        "total_droplets_in_cell": int(len(cell_df)),
                        "n_images": int(n_images),
                    }
                )
    return pd.DataFrame(records, columns=columns)


def plot_droplet_state_map(region_df: pd.DataFrame, output_path: Path, grid_cols: int = 12, grid_rows: int = 8) -> pd.DataFrame:
    plot_df = droplet_state_map_data(region_df, grid_cols=grid_cols, grid_rows=grid_rows)
    if plot_df.empty:
        return plot_df

    groups = list(plot_df["group"].dropna().astype(str).unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups) + 1.5, 4.4), squeeze=False)
    cmap = plt.get_cmap("plasma")
    max_fraction = float(plot_df["mean_droplet_fraction_per_image"].max()) if plot_df["mean_droplet_fraction_per_image"].notna().any() else 0.0

    for col_idx, group in enumerate(groups):
        ax = axes[0, col_idx]
        part = plot_df[(plot_df["group"].astype(str) == group) & (plot_df["total_droplets_in_cell"] > 0)].copy()
        if not part.empty:
            sizes = 25.0 + 1700.0 * part["mean_droplet_fraction_per_image"].fillna(0) / max(max_fraction, 1e-9)
            sc = ax.scatter(
                part["center_x_um"],
                part["center_y_um"],
                s=sizes,
                c=part["p_coal_ML"],
                cmap=cmap,
                vmin=0,
                vmax=1,
                alpha=0.78,
                linewidths=0.35,
                edgecolors="#555555",
            )
        ax.set_xlim(0, FIELD_WIDTH_UM)
        ax.set_ylim(FIELD_HEIGHT_UM, 0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{group}\nall images summarized", fontsize=11, fontweight="bold")
        ax.set_xlabel("x (um)")
        if col_idx == 0:
            ax.set_ylabel("y (um)")
        else:
            ax.set_ylabel("")
        ax.grid(alpha=0.18)
        for spine in ax.spines.values():
            spine.set_color("#B0B0B0")
            spine.set_linewidth(0.8)

    fig.suptitle("Droplet state map across all images", y=0.98, fontsize=13, fontweight="bold")
    cbar_ax = fig.add_axes([0.90, 0.24, 0.025, 0.52])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Mean coalescence probability", fontsize=9)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["0\nsmall/discrete", "0.5\ntransition", "1\nlarge/coalesced"])

    size_ax = fig.add_axes([0.16, 0.05, 0.46, 0.08])
    size_ax.axis("off")
    size_values = [0.01, 0.03, 0.06, 0.10]
    x_positions = np.linspace(0.12, 0.88, len(size_values))
    for x_pos, value in zip(x_positions, size_values):
        size_ax.scatter([x_pos], [0.58], s=25.0 + 1700.0 * value / max(max_fraction, 1e-9), color="#BDBDBD", edgecolors="#777777")
        size_ax.text(x_pos, 0.05, f"{value:.0%}", ha="center", va="center", fontsize=8)
    size_ax.text(0.0, 0.58, "marker_size = mean_droplet_fraction_per_image", ha="left", va="center", fontsize=8)

    plt.subplots_adjust(left=0.08, right=0.86, top=0.82, bottom=0.18, wspace=0.18)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return plot_df


def image_kinetic_state_data(image_results: pd.DataFrame, model_parameters: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = ["group", "image", "image_path", "q_global", "phi", "F_coal_ML", "CV_phi", "marker_size"]
    required = ["group", "image", "image_path", "phi", "F_coal_ML", "CV_phi"]
    if image_results.empty or any(column not in image_results for column in required):
        return pd.DataFrame(columns=columns)

    df = image_results[required].replace([np.inf, -np.inf], np.nan).dropna(subset=["phi", "F_coal_ML", "CV_phi"]).copy()
    if df.empty:
        return pd.DataFrame(columns=columns)
    q_global = np.nan
    if model_parameters is not None and not model_parameters.empty:
        q_global = model_parameters.iloc[0].to_dict().get("q_global", np.nan)
    df["q_global"] = q_global
    cv = df["CV_phi"].astype(float)
    if float(cv.max()) > float(cv.min()):
        df["marker_size"] = 45.0 + 185.0 * (cv - cv.min()) / (cv.max() - cv.min())
    else:
        df["marker_size"] = 110.0
    return df[columns]


def plot_image_kinetic_state_map(image_results: pd.DataFrame, model_parameters: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    plot_df = image_kinetic_state_data(image_results, model_parameters)
    if plot_df.empty:
        return plot_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    palette = sns.color_palette("bright", n_colors=max(2, plot_df["group"].nunique()))
    group_colors = {group: palette[idx] for idx, group in enumerate(plot_df["group"].dropna().astype(str).unique())}
    for group, part in plot_df.groupby("group"):
        ax.scatter(
            part["F_coal_ML"],
            part["phi"],
            s=part["marker_size"],
            color=group_colors[str(group)],
            alpha=0.58,
            edgecolors="#4F4F4F",
            linewidths=0.5,
            label=str(group),
        )
        ax.scatter(
            [part["F_coal_ML"].mean()],
            [part["phi"].mean()],
            s=260,
            color=group_colors[str(group)],
            edgecolors="#111111",
            linewidths=1.2,
            marker="D",
            label=f"{group} mean",
        )
        ax.text(part["F_coal_ML"].mean(), part["phi"].mean(), str(group), color=group_colors[str(group)], fontweight="bold", fontsize=10)

    x_min, x_max = plot_df["F_coal_ML"].min(), plot_df["F_coal_ML"].max()
    y_min, y_max = plot_df["phi"].min(), plot_df["phi"].max()
    x_pad = max((x_max - x_min) * 0.18, 0.03)
    y_pad = max((y_max - y_min) * 0.18, 0.03)
    ax.set_xlim(max(0, x_min - x_pad), min(1.05, x_max + x_pad))
    ax.set_ylim(max(0, y_min - y_pad), min(1.05, y_max + y_pad))
    ax.set_xlabel(r"$F_{\mathrm{coal,ML}}=\frac{\sum_i p_i A_i}{\sum_i A_i}$")
    ax.set_ylabel(r"$\phi=\frac{A_{\mathrm{droplet}}}{A_{\mathrm{image}}}$")
    q_global = plot_df["q_global"].replace([np.inf, -np.inf], np.nan).dropna()
    q_text = f"q_global = {float(q_global.iloc[0]):.3f}" if len(q_global) else "q_global unavailable"
    ax.set_title(f"Image-level dynamics parameter bubble map\n{q_text}", fontweight="bold")

    cv_values = np.nanpercentile(plot_df["CV_phi"], [20, 50, 80])
    handles = [
        ax.scatter([], [], s=45.0 + 185.0 * (value - plot_df["CV_phi"].min()) / max(plot_df["CV_phi"].max() - plot_df["CV_phi"].min(), 1e-9), color="#BDBDBD", edgecolors="#777777", label=f"{value:.2f}")
        for value in cv_values
    ]
    size_legend = ax.legend(handles=handles, title="Circle size\nCV_phi", loc="lower right", frameon=True)
    ax.add_artist(size_legend)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    if len(plot_df) >= 2:
        ax.annotate(
            "higher coverage\nstronger coalescence",
            xy=(x_max, y_max),
            xytext=(x_max - x_pad * 0.9, y_max - y_pad * 0.35),
            arrowprops={"arrowstyle": "-|>", "color": "#888888", "lw": 1.4},
            fontsize=8,
            ha="right",
        )
    ax.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return plot_df


def radar_plot_data(dynamics_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    axes = ["q_global", "phi", "F_coal_ML", "CV_phi"]
    if dynamics_input.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw = dynamics_input[["sample"] + axes].copy()
    norm = raw.copy()
    for axis in axes:
        values = raw[axis].replace([np.inf, -np.inf], np.nan)
        min_value = float(values.min()) if values.notna().any() else np.nan
        max_value = float(values.max()) if values.notna().any() else np.nan
        if np.isfinite(min_value) and np.isfinite(max_value) and max_value > min_value:
            norm[axis] = (values - min_value) / (max_value - min_value)
        elif np.isfinite(min_value):
            norm[axis] = 0.5
        else:
            norm[axis] = np.nan
    return raw, norm


def plot_gmm_classification_density(region_df: pd.DataFrame, model_parameters: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    curve_df = gmm_plot_data(region_df, model_parameters)
    if curve_df.empty:
        return curve_df

    params = model_parameters.iloc[0].to_dict()
    q_log = params.get("q_ML_log", np.nan)
    values = region_df["ln_Ai_A0"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 5:
        return curve_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.6, 4.8))
    plt.hist(values, bins=45, density=True, color="#D9D9D9", edgecolor="#8A8A8A", alpha=0.72, label="Actual droplets")

    for group, part in region_df.groupby("group"):
        group_values = part["ln_Ai_A0"].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if len(group_values) < 3:
            continue
        counts, edges = np.histogram(group_values, bins=45, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        plt.plot(centers, counts, lw=1.4, alpha=0.8, label=f"{group} density")

    plt.plot(curve_df["x_ln_Ai_A0"], curve_df["small_discrete_density"], color="#3366CC", lw=2.2, label="GMM: small/discrete")
    plt.plot(curve_df["x_ln_Ai_A0"], curve_df["large_coalesced_density"], color="#CC3333", lw=2.2, label="GMM: large/coalesced")
    plt.plot(curve_df["x_ln_Ai_A0"], curve_df["gmm_total_density"], color="#222222", lw=1.8, linestyle="--", label="GMM total")

    ymax = float(np.nanmax(curve_df[["gmm_total_density", "small_discrete_density", "large_coalesced_density"]].to_numpy()))
    if np.isfinite(q_log):
        plt.axvline(q_log, color="#7B2CBF", lw=1.8, linestyle="--", label="q")
        plt.text(q_log, ymax * 0.92, "q", color="#7B2CBF", ha="center", va="bottom", fontsize=11)
    plt.text(curve_df["x_ln_Ai_A0"].min(), ymax * 0.76, "small/discrete droplets", color="#3366CC", ha="left")
    plt.text(curve_df["x_ln_Ai_A0"].max(), ymax * 0.76, "large/coalesced droplets", color="#CC3333", ha="right")
    plt.xlabel("x_i = ln(A_i / A_0)")
    plt.ylabel("Probability density")
    plt.title("Droplet normalized area distribution and GMM fitting")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return curve_df


def plot_coalescence_probability_curve(region_df: pd.DataFrame, model_parameters: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    curve_df = gmm_plot_data(region_df, model_parameters)
    if curve_df.empty:
        return curve_df

    params = model_parameters.iloc[0].to_dict()
    q_log = params.get("q_ML_log", np.nan)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.6, 4.8))
    plt.plot(curve_df["x_ln_Ai_A0"], curve_df["p_coal_ML"], color="#335CFF", lw=2.4, label="p_i = P(large/coalesced | x_i)")

    scatter_df = region_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ln_Ai_A0", "p_coal_ML"])
    if not scatter_df.empty:
        sns.scatterplot(data=scatter_df, x="ln_Ai_A0", y="p_coal_ML", hue="group", s=22, alpha=0.5, edgecolor=None)

    if np.isfinite(q_log):
        plt.axvline(q_log, color="#7B2CBF", lw=1.8, linestyle="--", label="q")
        plt.axhline(0.5, color="#777777", lw=1.1, linestyle="--")
        plt.text(q_log, 0.94, "q", color="#7B2CBF", ha="center", va="bottom", fontsize=11)
    plt.ylim(-0.03, 1.03)
    plt.xlabel("x_i = ln(A_i / A_0)")
    plt.ylabel("p_i")
    plt.title("Coalescence probability as a function of normalized area")
    plt.text(curve_df["x_ln_Ai_A0"].min(), 0.18, "small/discrete\np_i ≈ 0", color="#3366CC", ha="left")
    plt.text(curve_df["x_ln_Ai_A0"].max(), 0.78, "large/coalesced\np_i ≈ 1", color="#CC3333", ha="right")
    plt.text(q_log if np.isfinite(q_log) else curve_df["x_ln_Ai_A0"].median(), 0.55, "transition\np_i ≈ 0.5", color="#7B2CBF", ha="center")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return curve_df[["x_ln_Ai_A0", "p_coal_ML", "q_global_log", "q_global"]].copy()


def plot_parameter_fingerprint_radar(dynamics_input: pd.DataFrame, output_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, norm = radar_plot_data(dynamics_input)
    if norm.empty:
        return raw, norm

    axes = ["q_global", "phi", "F_coal_ML", "CV_phi"]
    labels = ["q", "phi", "Fcoal_ML", "CV_phi"]
    angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
    angles += angles[:1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"polar": True})
    for _, row in norm.iterrows():
        values = [row[axis] for axis in axes]
        values += values[:1]
        ax.plot(angles, values, marker="o", linewidth=2, label=str(row["sample"]))
        ax.fill(angles, values, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_title("ML output parameter fingerprint", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.24, 1.12), frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return raw, norm


def export_figure_table(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})


def write_ml_figure_data(
    output_path: Path,
    gmm_density_df: pd.DataFrame,
    actual_density_df: pd.DataFrame,
    area_histogram_df: pd.DataFrame,
    image_state_df: pd.DataFrame,
    heatmap_values_df: pd.DataFrame,
    heatmap_corr_df: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        export_figure_table(
            gmm_density_df,
            {
                "x_ln_Ai_A0": "x_i = ln(A_i / A_0)",
                "small_discrete_density": "GMM: small/discrete",
                "large_coalesced_density": "GMM: large/coalesced",
                "gmm_total_density": "GMM total",
                "p_coal_ML": "p_i",
                "q_global_log": "q",
            },
        ).to_excel(writer, sheet_name="A_GMM_density_curves", index=False)
        export_figure_table(
            actual_density_df,
            {
                "group": "Sample",
                "x_ln_Ai_A0": "x_i = ln(A_i / A_0)",
                "probability_density": "Probability density",
            },
        ).to_excel(writer, sheet_name="A_actual_hist_density", index=False)
        export_figure_table(
            area_histogram_df,
            {
                "group": "Sample",
                "A_i_um2": "Droplet area A_i (um2)",
                "droplet_count": "Droplet count",
                "probability_density": "Probability density",
            },
        ).to_excel(writer, sheet_name="area_distribution_hist", index=False)
        export_figure_table(
            image_state_df,
            {
                "group": "Sample",
                "image": "Image",
                "image_path": "Image path",
                "phi": "phi",
                "F_coal_ML": "F_coal_ML",
                "CV_phi": "CV_phi",
                "marker_size": "Circle size",
            },
        ).to_excel(writer, sheet_name="image_dynamics_bubbles", index=False)
        export_figure_table(
            heatmap_values_df,
            {"group": "Sample", "image": "Image", "image_path": "Image path"},
        ).to_excel(writer, sheet_name="heatmap_core_values", index=False)
        export_figure_table(
            heatmap_corr_df,
            {"parameter": "Parameter"},
        ).to_excel(writer, sheet_name="heatmap_core_correlation", index=False)


def plot_ml_factor_boxplots(image_results: pd.DataFrame, output_path: Path) -> None:
    metrics = ["phi", "F_coal_ML", "CV_phi", "Phi_ML"]
    available = [metric for metric in metrics if metric in image_results]
    if image_results.empty or len(available) == 0 or "group" not in image_results:
        return

    plot_df = image_results.melt(id_vars=["group"], value_vars=available, var_name="metric", value_name="value")
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    if plot_df.empty:
        return

    g = sns.catplot(
        data=plot_df,
        x="group",
        y="value",
        col="metric",
        kind="box",
        sharey=False,
        col_wrap=2,
        height=3.4,
        aspect=1.2,
        color="#88CCEE",
    )
    g.set_axis_labels("", "")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=30)
    g.fig.suptitle("Image-level ML film factors", y=1.02)
    g.fig.tight_layout()
    g.fig.savefig(output_path, dpi=300)
    plt.close(g.fig)


CORE_HEATMAP_COLUMNS = {
    "N": "n_droplets",
    "phi": "phi",
    "A": "mean_area_um2",
    "F_coal": "F_coal_ML",
    "CV_phi": "CV_phi",
    "d_NN": "mean_nearest_neighbor_distance_um",
    "C": "mean_circularity",
}


def core_heatmap_values(image_df: pd.DataFrame) -> pd.DataFrame:
    if image_df.empty:
        return pd.DataFrame(columns=["group", "image"] + list(CORE_HEATMAP_COLUMNS.keys()))

    metadata = [column for column in ["group", "image", "image_path"] if column in image_df]
    values = image_df[metadata].copy()
    for label, source_column in CORE_HEATMAP_COLUMNS.items():
        values[label] = image_df[source_column] if source_column in image_df else np.nan
    return values.replace([np.inf, -np.inf], np.nan)


def plot_numeric_heatmap(image_df: pd.DataFrame, output_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = core_heatmap_values(image_df)
    numeric = values[list(CORE_HEATMAP_COLUMNS.keys())].dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        return values, pd.DataFrame()

    corr = numeric.corr(numeric_only=True)
    labels = list(corr.columns)
    plt.figure(figsize=(7.2, 6.2))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.78},
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Core droplet-kinetic parameter correlation")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return values, corr.reset_index().rename(columns={"index": "parameter"})


def write_excel(
    output_path: Path,
    dynamics_input: pd.DataFrame,
    region_df: pd.DataFrame,
    image_results: pd.DataFrame,
    sample_summary: pd.DataFrame,
    model_parameters: pd.DataFrame,
    a0_candidates: pd.DataFrame,
    all_image_df: pd.DataFrame | None = None,
    excluded_image_df: pd.DataFrame | None = None,
    quality_summary_df: pd.DataFrame | None = None,
) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        dynamics_input.to_excel(writer, sheet_name="dynamics_input", index=False)
        image_results.to_excel(writer, sheet_name="image_results", index=False)
        sample_summary.to_excel(writer, sheet_name="sample_summary", index=False)
        model_parameters.to_excel(writer, sheet_name="model_parameters", index=False)
        region_df.to_excel(writer, sheet_name="droplet_regions", index=False)
        if all_image_df is not None:
            all_image_df.to_excel(writer, sheet_name="all_image_summary", index=False)
        if excluded_image_df is not None:
            excluded_image_df.to_excel(writer, sheet_name="excluded_images", index=False)
        if quality_summary_df is not None:
            quality_summary_df.to_excel(writer, sheet_name="quality_filter_summary", index=False)
        a0_candidates.to_excel(writer, sheet_name="A0_candidates", index=False)


def batch_worker_count(n_images: int) -> int:
    if n_images <= 1 or not BATCH_PARALLEL_PROCESSING:
        return 1
    if gpu_acceleration_available():
        return 1
    return max(1, min(MAX_BATCH_WORKERS, n_images))


def process_batch_image(payload: tuple[int, int, str, str, str, str, str, str, str, SegmentationParams]) -> dict:
    i, total, path_text, group, group_root_text, output_stem_text, overlay_dir_text, gray_dir_text, binary_dir_text, params = payload
    path = Path(path_text)
    group_root = Path(group_root_text)
    output_stem = Path(output_stem_text)
    overlay_dir = Path(overlay_dir_text)
    gray_dir = Path(gray_dir_text)
    binary_dir = Path(binary_dir_text)

    try:
        image_rgb = read_image(path)
        seg = background_correct_and_segment(image_rgb, params)
        region_df = extract_region_features(path, group, seg, group_root, params)
        summary = image_summary(path, group, seg, region_df)

        save_gray_and_binary_images(seg, gray_dir / f"{output_stem}.png", binary_dir / f"{output_stem}.png")
        save_segmentation_overlay(seg, region_df, overlay_dir / f"{output_stem}.png")

        return {
            "ok": True,
            "index": i,
            "total": total,
            "image_name": path.name,
            "group": group,
            "n_droplets": int(len(region_df)),
            "region_records": region_df.to_dict("records"),
            "summary": summary,
        }
    except Exception as exc:
        return {
            "ok": False,
            "index": i,
            "total": total,
            "image_path": str(path),
            "error": repr(exc),
        }


def choose_single_image_file() -> Path:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a single image for recognition",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        raise SystemExit("No image selected. Single-image recognition cancelled.")
    return Path(path).expanduser().resolve()


def parse_optional_output_dir(args: list[str]) -> Path | None:
    for flag in ("--output", "--single-output"):
        if flag in args:
            index = args.index(flag)
            if len(args) <= index + 1:
                raise SystemExit(f"{flag} requires an output folder argument.")
            return Path(args[index + 1]).expanduser().resolve()
    return None


def parse_single_image_path(args: list[str]) -> Path:
    for flag in ("--single", "--single-image"):
        if flag in args:
            index = args.index(flag)
            if len(args) > index + 1:
                candidate_text = args[index + 1]
                if not candidate_text.startswith("--"):
                    candidate = Path(candidate_text).expanduser().resolve()
                    if candidate.exists() and candidate.is_file():
                        return candidate
                    raise SystemExit(f"Single-image path does not exist or is not a file: {candidate}")
            return choose_single_image_file()
    raise SystemExit("Missing --single or --single-image argument.")


def run_single_image_recognition(
    image_path: Path,
    params: SegmentationParams | None = None,
    output_dir: Path | None = None,
) -> Path:
    params = params or load_segmentation_params()
    image_path = image_path.expanduser().resolve()
    if not image_path.exists() or not image_path.is_file():
        raise SystemExit(f"Single-image path does not exist or is not a file: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise SystemExit(f"Unsupported image format: {image_path.suffix}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir or (image_path.parent / f"single_image_recognition_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_rgb = read_image(image_path)
    seg = background_correct_and_segment(image_rgb, params)
    region_df = extract_region_features(image_path, image_path.parent.name, seg, image_path.parent, params)

    stem = image_path.stem
    gray_path = output_dir / f"{stem}_gray_corrected.png"
    binary_path = output_dir / f"{stem}_binary_mask.png"
    overlay_path = output_dir / f"{stem}_segmentation_overlay.png"

    save_gray_and_binary_images(seg, gray_path, binary_path)
    save_segmentation_overlay(seg, region_df, overlay_path)

    print("Single-image recognition complete.")
    print(f"Input image: {image_path}")
    print(f"droplets={len(region_df)}")
    print(f"Corrected grayscale image: {gray_path}")
    print(f"Binary mask: {binary_path}")
    print(f"Segmentation overlay: {overlay_path}")
    return output_dir


def main(
    root: Path | None = None,
    params: SegmentationParams | None = None,
    group_folders: list[GroupFolder] | None = None,
) -> None:
    params = params or load_segmentation_params()
    if group_folders is None:
        root = root or choose_root_folder()
        group_folders = [GroupFolder(group=root.name, folder=root)]

    image_records = collect_group_images(group_folders)
    if not image_records:
        folders = ", ".join(str(item.folder) for item in group_folders)
        raise SystemExit(f"No image files were found in the selected folder(s): {folders}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = common_output_root(group_folders)
    output_dir = output_root / f"droplet_ml_results_{timestamp}"
    overlay_dir = output_dir / "segmentation_overlays"
    gray_dir = output_dir / "gray_corrected_images"
    binary_dir = output_dir / "binary_masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_segmentation_params(params, output_dir / "segmentation_params_used.json")
    pd.DataFrame(
        [{"group": item.group, "folder": str(item.folder)} for item in group_folders]
    ).to_csv(output_dir / "group_folders_used.csv", index=False, encoding="utf-8-sig")

    all_regions = []
    summaries = []
    errors = []

    group_text = "; ".join(f"{item.group}: {item.folder}" for item in group_folders)
    print(f"Selected experimental groups: {group_text}")
    print(f"Found {len(image_records)} image(s). Starting processing...")
    print(gpu_status_text())

    workers = batch_worker_count(len(image_records))
    if workers > 1:
        print(f"CPU parallel batch processing enabled: {workers} processes")
    else:
        print("Batch mode: single process")

    def collect_result(result: dict) -> None:
        if result["ok"]:
            n_droplets = int(result["n_droplets"])
            all_regions.append(pd.DataFrame(result["region_records"]))
            summaries.append(result["summary"])
            if n_droplets < MIN_VALID_DROPLETS_PER_IMAGE:
                print(
                    f"[{result['index']}/{result['total']}] Done: {result['group']}/{result['image_name']}, droplets={n_droplets} "
                    f"-> flagged as abnormal and excluded from final analysis"
                )
            else:
                print(f"[{result['index']}/{result['total']}] Done: {result['group']}/{result['image_name']}, droplets={n_droplets}")
        else:
            errors.append({"image_path": result["image_path"], "error": result["error"]})
            print(f"[{result['index']}/{result['total']}] Failed: {result['image_path']} -> {result['error']}")

    payloads = [
        (
            i,
            len(image_records),
            str(path),
            group,
            str(group_root),
            str(output_stem),
            str(overlay_dir),
            str(gray_dir),
            str(binary_dir),
            params,
        )
        for i, (path, group, group_root, output_stem) in enumerate(image_records, start=1)
    ]

    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_index = {executor.submit(process_batch_image, payload): payload[0] for payload in payloads}
                for future in as_completed(future_to_index):
                    collect_result(future.result())
        except PermissionError as exc:
            print(f"CPU parallel batch initialization failed; falling back to single-process mode: {exc}")
            for payload in payloads:
                collect_result(process_batch_image(payload))
    else:
        for payload in payloads:
            collect_result(process_batch_image(payload))

    raw_region_df = pd.concat(all_regions, ignore_index=True) if all_regions else pd.DataFrame()
    raw_image_df = pd.DataFrame(summaries)
    region_df, image_df, excluded_image_df, all_image_df = filter_valid_images_by_quality(raw_region_df, raw_image_df)
    quality_summary_df = quality_filter_summary(all_image_df, image_df, excluded_image_df, failed_count=len(errors))
    quality_summary_df.to_csv(output_dir / "quality_filter_summary.csv", index=False, encoding="utf-8-sig")

    if not excluded_image_df.empty:
        excluded_image_df.to_csv(output_dir / "excluded_images_by_quality.csv", index=False, encoding="utf-8-sig")

    print(
        "Quality-filter summary: "
        f"processed {len(all_image_df)}; "
        f"included in final analysis {len(image_df)}; "
        f"excluded by quality filter {len(excluded_image_df)}; "
        f"failed {len(errors)}."
    )
    if not excluded_image_df.empty:
        print("Excluded-image details: excluded_images_by_quality.csv; quality-filter summary: quality_filter_summary.csv")

    if region_df.empty:
        raise SystemExit(
            "No droplet regions remain after quality filtering. Check the gray threshold offset or adjust "
            "MIN_VALID_DROPLETS_PER_IMAGE."
        )

    a0, a0_candidates, a0_message = estimate_a0(region_df)
    region_df, image_results, model_parameters, q_boot = fit_unified_ml_model(region_df, image_df, a0)
    model_parameters["A0_message"] = a0_message
    sample_summary = build_sample_summary(image_results, model_parameters)
    dynamics_input = build_dynamics_input_table(image_results, model_parameters)
    dynamics_input.to_excel(output_dir / "dynamics_input.xlsx", sheet_name="dynamics_input", index=False)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    heatmap_values_df, heatmap_corr_df = plot_numeric_heatmap(image_results, figures_dir / "numeric_matrix_heatmap.png")
    gmm_density_df = plot_gmm_classification_density(region_df, model_parameters, figures_dir / "A_GMM_area_density.png")
    area_histogram_df = plot_area_distribution_histogram(region_df, figures_dir / "area_distribution_histogram.png")
    image_state_df = plot_image_kinetic_state_map(image_results, model_parameters, figures_dir / "image_dynamics_parameter_bubble_map.png")
    actual_density_df = actual_distribution_data(region_df)
    figure_data_path = output_dir / "ml_figure_data.xlsx"
    write_ml_figure_data(
        figure_data_path,
        gmm_density_df,
        actual_density_df,
        area_histogram_df,
        image_state_df,
        heatmap_values_df,
        heatmap_corr_df,
    )

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "processing_errors.csv", index=False, encoding="utf-8-sig")

    excel_path = output_dir / "droplet_ml_results.xlsx"
    write_excel(
        excel_path,
        dynamics_input,
        region_df,
        image_results,
        sample_summary,
        model_parameters,
        a0_candidates,
        all_image_df,
        excluded_image_df,
        quality_summary_df,
    )

    print("\nAnalysis complete.")
    print(f"Output directory: {output_dir}")
    print(f"Excel: {excel_path}")
    print(f"Dynamics-model input table: {output_dir / 'dynamics_input.xlsx'}")
    print(f"ML figure-data workbook: {figure_data_path}")
    print(a0_message)
    model_row = model_parameters.iloc[0].to_dict() if not model_parameters.empty else {}
    print(f"q_ML status: {model_row.get('q_status')}")
    print(f"q_ML note: {model_row.get('q_reason')}")
    if np.isfinite(model_row.get("q_global", np.nan)):
        print(
            f"q_global={model_row['q_global']:.4f} "
            f"(ln q={model_row['q_ML_log']:.4f}), "
            f"95% CI=({model_row['q_ci95_area_multiple_low']:.4f}, {model_row['q_ci95_area_multiple_high']:.4f})"
        )


if __name__ == "__main__":
    try:
        if "--single" in sys.argv or "--single-image" in sys.argv:
            single_image = parse_single_image_path(sys.argv[1:])
            single_output_dir = parse_optional_output_dir(sys.argv[1:])
            run_single_image_recognition(single_image, output_dir=single_output_dir)
        elif "--batch" in sys.argv:
            batch_index = sys.argv.index("--batch")
            batch_root = None
            if len(sys.argv) > batch_index + 1:
                candidate = Path(sys.argv[batch_index + 1]).expanduser().resolve()
                if candidate.exists() and candidate.is_dir():
                    batch_root = candidate
            main(batch_root)
        else:
            tune_image = None
            initial_batch_folder = None
            for arg in sys.argv[1:]:
                if arg.startswith("--"):
                    continue
                candidate = Path(arg).expanduser().resolve()
                if candidate.exists() and candidate.is_file():
                    tune_image = candidate
                    break
                if candidate.exists() and candidate.is_dir():
                    initial_batch_folder = candidate

            should_run, selected_groups, tuned_params = run_parameter_tuner(tune_image, initial_batch_folder)
            if should_run and selected_groups:
                main(params=tuned_params, group_folders=selected_groups)
            else:
                print("Parameter tuner closed; batch processing was not started.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
