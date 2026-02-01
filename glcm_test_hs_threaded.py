import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import os
import sys
import concurrent.futures
import random
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

possible_paths = [
    os.path.join(SCRIPT_DIR, 'bananadataset', 'dataset', 'data'),
    os.path.join(SCRIPT_DIR, 'Assets', 'bananadataset', 'dataset', 'data'),
    os.path.abspath('Assets/bananadataset/dataset/data')
]
BASE_PATH = next((p for p in possible_paths if os.path.exists(p)), 'Assets/bananadataset/dataset/data')

CATEGORIES = ['unripe', 'ripe', 'overripe']
SPLITS = ['train', 'test', 'valid']

# Limit wizualizacji (ile masek pokazać w galerii)
VIS_LIMIT = 50


# --- SEGMENTACJA: K-MEANS (kanały HUE + SATURATION) ---
def get_mask_kmeans_hs(img):
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]  # Barwa
        s_channel = hsv[:, :, 1]  # Nasycenie

        features = np.dstack((h_channel, s_channel)).reshape((-1, 2)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(features, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        mask = labels.reshape(img.shape[:2]).astype(np.uint8) * 255

        mean_s_0 = np.mean(s_channel[mask == 0])
        mean_s_255 = np.mean(s_channel[mask == 255])

        if mean_s_0 > mean_s_255:
            mask = cv2.bitwise_not(mask)

        # Czyszczenie, czasami pomaga z dziurami w bananie
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(mask)
            cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)
            return clean_mask

        return mask
    except:
        return None


# --- CECHY GLCM ---
def calculate_features(img, mask):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        glcm = graycomatrix(masked_gray, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True,
                            normed=True)
        glcm[0, :, :, :] = 0;
        glcm[:, 0, :, :] = 0
        if np.sum(glcm) > 0: glcm /= np.sum(glcm)

        feats = {
            "Contrast": np.mean(graycoprops(glcm, 'contrast')),
            "Correlation": np.mean(graycoprops(glcm, 'correlation')),
            "Energy": np.mean(graycoprops(glcm, 'energy')),
            "Homogeneity": np.mean(graycoprops(glcm, 'homogeneity'))
        }
        viz = np.log(np.mean(glcm, axis=(2, 3)) + 1e-9)
        return feats, masked_gray, viz
    except:
        return None, None, None


# --- WORKER (wątki) ---
def process_category_worker(category):
    category_stats = {m: [] for m in ['Contrast', 'Correlation', 'Energy', 'Homogeneity']}
    sample_vis_data = None
    gallery_thumbnails = []

    total_images_processed = 0
    print(f"🚀 [PID {os.getpid()}] START: {category.upper()}", flush=True)

    for split in SPLITS:
        folder = os.path.join(BASE_PATH, split, category)
        if not os.path.exists(folder): continue

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))])
        count = len(files)
        sample_idx = random.randint(0, count - 1) if count > 0 else -1

        for i, fname in enumerate(files):
            if i > 0 and i % 50 == 0:
                print(f"      ⚙️ [PID {os.getpid()}] {split}/{category}: {i}/{count}...", flush=True)

            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is None: continue

            mask = get_mask_kmeans_hs(img)
            if mask is None: continue

            feats, masked_gray, viz = calculate_features(img, mask)
            if feats is None: continue

            for k, v in feats.items():
                category_stats[k].append(v)

            total_images_processed += 1

            if len(gallery_thumbnails) < VIS_LIMIT:
                thumb = cv2.resize(masked_gray, (100, 100), interpolation=cv2.INTER_AREA)
                gallery_thumbnails.append(thumb)

            if sample_vis_data is None or i == sample_idx:
                img_s = cv2.resize(img, (300, 300))
                mask_s = cv2.resize(mask, (300, 300), interpolation=cv2.INTER_NEAREST)
                masked_s = cv2.resize(masked_gray, (300, 300))
                info_txt = f"K-Means (H+S)\n({split}/{fname})"
                sample_vis_data = (img_s, mask_s, masked_s, viz, info_txt)

    print(f"✅ [PID {os.getpid()}] KONIEC: {category.upper()}. Razem: {total_images_processed}", flush=True)
    return category, category_stats, sample_vis_data, gallery_thumbnails


# --- WIZUALIZACJA ---
def calculate_outliers_count(data):
    if not data: return 0
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    return len([x for x in data if x < q1 - 1.5 * iqr or x > q3 + 1.5 * iqr])


def plot_category_dashboard(category, stats, sample_data):
    if sample_data is None: return
    img, final_mask, input_glcm, heatmap, info_txt = sample_data

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"STATYSTYKI: {category.upper()} (Train+Test+Valid)", fontsize=18, fontweight='bold', color='darkblue')

    ax1 = fig.add_subplot(3, 4, 1);
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
    ax1.set_title("Losowy Oryginał")
    ax2 = fig.add_subplot(3, 4, 2);
    ax2.imshow(final_mask, cmap='gray');
    ax2.set_title(f"Maska\n{info_txt}", fontsize=9)
    ax3 = fig.add_subplot(3, 4, 3);
    ax3.imshow(input_glcm, cmap='gray');
    ax3.set_title("Input GLCM")
    ax4 = fig.add_subplot(3, 4, 4);
    ax4.imshow(heatmap, cmap='viridis');
    ax4.set_title("Heatmapa GLCM")

    ax_txt = fig.add_subplot(3, 4, 5);
    ax_txt.axis('off')
    n_samples = len(stats['Contrast'])
    txt = f"Łącznie zdjęć: {n_samples}\nOUTLIERS (Anomalie):\n"
    for k, v in stats.items():
        cnt = calculate_outliers_count(v)
        perc = (cnt / n_samples * 100) if n_samples > 0 else 0
        txt += f"{k}: {cnt} ({perc:.1f}%)\n"
    ax_txt.text(0, 0.5, txt, fontsize=12, va='center')

    cols = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
    metrics = ['Contrast', 'Correlation', 'Energy', 'Homogeneity']
    for i, m in enumerate(metrics):
        ax = fig.add_subplot(3, 4, 9 + i)
        vals = stats[m]
        if vals:
            ax.hist(vals, bins=50, color=cols[i], edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(vals), color='black', linestyle='--')
            ax.set_title(f"Rozkład: {m}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(f"📊 [1/2] Dashboard statystyk dla {category}. Zamknij okno...")
    plt.show()


# --- WIZUALIZACJA - MASKI ---
def plot_mask_gallery(category, thumbnails):
    total = len(thumbnails)
    if total == 0: return

    cols = 10
    rows = math.ceil(total / cols)

    print(f"🖼️ [2/2] Galeria masek dla {category}...")

    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 1.5))
    fig.suptitle(f"GALERIA PRÓBEK: {category.upper()} (Metoda Hue+Sat)", fontsize=16, fontweight='bold', y=1.0)

    axes_flat = axes.flatten() if total > 1 else [axes]
    frame_color = {'unripe': 'green', 'ripe': 'gold', 'overripe': 'red'}.get(category, 'blue')

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < total:
            ax.imshow(thumbnails[i], cmap='gray')
            for spine in ax.spines.values():
                spine.set_edgecolor(frame_color)
                spine.set_linewidth(2)
            ax.set_xticks([]);
            ax.set_yticks([])
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# --- PODSUMOWANIA KOŃCOWE ---

def plot_final_comparison(all_data_aggregated):
    print("\n📊 Generowanie zestawienia Z OUTLIERAMI...")
    metrics_config = [
        ('Contrast', 'Szorstkość', '#3498db'),
        ('Homogeneity', 'Gładkość', '#f1c40f'),
        ('Correlation', 'Liniowość', '#e74c3c'),
        ('Energy', 'Jednolitość', '#2ecc71')
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PODSUMOWANIE (Z Outlierami)', fontsize=16, fontweight='bold')
    axes_flat = axes.flatten()
    box_colors = ['#ADD8E6', '#90EE90', '#FA8072']

    for idx, (metric, desc, _) in enumerate(metrics_config):
        ax = axes_flat[idx]
        data_to_plot = [all_data_aggregated[cat][metric] for cat in CATEGORIES]
        if not any(data_to_plot): continue
        bplot = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=[c.upper() for c in CATEGORIES], notch=True)
        for patch, c in zip(bplot['boxes'], box_colors): patch.set_facecolor(c)
        ax.set_title(f"{metric.upper()}", fontweight='bold')
        ax.set_xlabel(desc, style='italic', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def plot_final_comparison_no_outliers(all_data_aggregated):
    print("\n📊 Generowanie zestawienia BEZ OUTLIERÓW (Zoom na medianę)...")
    metrics_config = [
        ('Contrast', 'Szorstkość', '#3498db'),
        ('Homogeneity', 'Gładkość', '#f1c40f'),
        ('Correlation', 'Liniowość', '#e74c3c'),
        ('Energy', 'Jednolitość', '#2ecc71')
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PODSUMOWANIE (Bez Outlierów)', fontsize=16, fontweight='bold', color='darkred')
    axes_flat = axes.flatten()
    box_colors = ['#ADD8E6', '#90EE90', '#FA8072']

    for idx, (metric, desc, _) in enumerate(metrics_config):
        ax = axes_flat[idx]
        data_to_plot = [all_data_aggregated[cat][metric] for cat in CATEGORIES]
        if not any(data_to_plot): continue

        # --- KLUCZOWA ZMIANA: showfliers=False ---
        bplot = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=[c.upper() for c in CATEGORIES], notch=True,
                           showfliers=False)

        for patch, c in zip(bplot['boxes'], box_colors): patch.set_facecolor(c)
        ax.set_title(f"{metric.upper()}", fontweight='bold')
        ax.set_xlabel(desc, style='italic', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


# --- MAIN ---
def main():
    print(f"--- START ANALIZY (METODA HUE+SAT) ---")
    print(f"Ścieżka: {BASE_PATH}")

    aggregated_data = {cat: {m: [] for m in ['Contrast', 'Correlation', 'Energy', 'Homogeneity']} for cat in CATEGORIES}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_cat = {executor.submit(process_category_worker, cat): cat for cat in CATEGORIES}

        for future in concurrent.futures.as_completed(future_to_cat):
            cat, stats, sample_vis, gallery = future.result()
            aggregated_data[cat] = stats
            plot_category_dashboard(cat, stats, sample_vis)
            plot_mask_gallery(cat, gallery)

    # Dwa podsumowania na koniec
    plot_final_comparison(aggregated_data)
    plot_final_comparison_no_outliers(aggregated_data)


if __name__ == "__main__":
    main()