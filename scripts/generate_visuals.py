import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
from pathlib import Path


def generate_dashboard(
    image_path="data/processed/Week1_150607_B02_s1.tif",
    mask_path="output/masks.npy",
    features_path="output/cell_features.parquet",
    output_image="assets/results_summary.png"
):
    """
    Generates a "30-Second Review" dashboard showing:
    1. Raw Input Image (RGB Composite)
    2. Segmented Masks
    3. Robust Z-Score Analysis of Outliers
    """

    # Ensure assets dir exists
    Path(output_image).parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Data or Gen Synthetic
    if not Path(image_path).exists() or not Path(mask_path).exists() or not Path(features_path).exists():
        print("Real data artifacts not found. Generating SYNTHETIC demo data for visualization...")
        # Create synthetic 3-channel image with spots
        from skimage.draw import disk
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        masks = np.zeros((512, 512), dtype=np.uint16)

        rng = np.random.default_rng(42)
        n_cells = 50

        # Create a dataframe for features
        features = []

        for i in range(1, n_cells + 1):
            r = rng.integers(10, 30)
            cx, cy = rng.integers(r, 512-r, 2)

            # Draw Mask
            rr, cc = disk((cy, cx), r, shape=img.shape[:2])
            masks[rr, cc] = i

            # Draw Channels
            # Ch0 (Nuclei): High intensity
            img[rr, cc, 0] = rng.integers(100, 200)
            # Ch1 (Tubulin): Med intensity
            img[rr, cc, 1] = rng.integers(50, 150)
            # Ch2 (Actin): Variable (create some outliers)
            if i < 5:  # 5 Outliers
                val = rng.integers(220, 255)  # Bright
            else:
                val = rng.integers(50, 120)  # Normal
            img[rr, cc, 2] = val

            features.append({
                'label': i,
                # Approximate Z-score
                'mean_intensity_Ch2_norm': (val - 85) / 20.0,
                'circularity_norm': rng.normal(0, 1)
            })

        df = pd.DataFrame(features)

        # Save temp files so the rest of the logic holds? No, just use variables.
        # But we need to transpose img back to (Y, X, C) if we were loading it.
        # Here it is (Y, X, C) already.
    else:
        # Load Real Data
        img = tifffile.imread(image_path)
        img = np.transpose(img, (1, 2, 0))
        masks = np.load(mask_path)
        df = pd.read_parquet(features_path)

    # Normalize for display (0-1 range)
    img_disp = img.astype(float)
    for c in range(3):
        p_low, p_high = np.percentile(img_disp[..., c], (2, 99.8))
        img_disp[..., c] = np.clip(
            (img_disp[..., c] - p_low) / (p_high - p_low), 0, 1)

    # Masks
    masks = np.load(mask_path)

    # Features
    df = pd.read_parquet(features_path)

    # 2. Setup Plot
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    # A. Raw Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_disp)
    ax1.set_title("1. Neural Perception (Raw Input)",
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    # B. Segmentation
    ax2 = fig.add_subplot(gs[0, 1])
    # Create random colormap for masks
    import matplotlib.colors as mcolors
    unique_labels = np.unique(masks)
    colors = np.random.rand(len(unique_labels), 3)
    colors[0] = [0, 0, 0]  # Background black
    cmap = mcolors.ListedColormap(colors)

    ax2.imshow(masks, cmap=cmap)
    ax2.set_title(
        f"2. Vision Engine ({len(df)} Cells)", fontsize=14, fontweight='bold')
    ax2.axis('off')

    # C. Symbolic Reasoning (Z-Scores)
    ax3 = fig.add_subplot(gs[0, 2])

    # Feature to plot: Actin (Ch2) Normalization
    x_col = 'mean_intensity_Ch2_norm'
    y_col = 'circularity_norm'  # Just to spread them out

    # Highlight Outliers
    threshold = 3.0
    outliers = df[df[x_col].abs() > threshold]
    inliers = df[df[x_col].abs() <= threshold]

    sns.scatterplot(data=inliers, x=x_col, y=y_col, ax=ax3,
                    color='gray', alpha=0.5, label='Normal Population')
    sns.scatterplot(data=outliers, x=x_col, y=y_col, ax=ax3, color='red',
                    s=100, marker='X', label=f'Outliers (n={len(outliers)})')

    # Add Robust Median Line (0) and Thresholds (+/- 3)
    ax3.axvline(0, color='blue', linestyle='--',
                alpha=0.3, label='Population Median')
    ax3.axvline(3, color='red', linestyle=':', label='Robust Threshold (+3.0)')
    ax3.axvline(-3, color='red', linestyle=':')

    ax3.set_title("3. Symbolic Reasoning (Outlier Detection)",
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel("Actin Intensity (Robust Z-Score)")
    ax3.set_ylabel("Circularity (Robust Z-Score)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Main Title
    fig.suptitle(
        f"Bio-Oracle: Neuro-Symbolic Agentic AI Results (Sample: {Path(image_path).stem})", fontsize=16)

    # Save
    plt.savefig(output_image, dpi=150)
    print(f"Dashboard saved to {output_image}")


if __name__ == "__main__":
    generate_dashboard()
