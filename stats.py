import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def process_folders_in_batches(base_path='.', batch_size=100):
    # 1. Map all images to their parent folders
    folder_map = {}
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    print("Scanning directories...")
    for root, _, files in os.walk(base_path):
        images = [os.path.join(root, f) for f in files if f.lower().endswith(valid_extensions)]
        if images:
            folder_map[root] = images

    if not folder_map:
        print("No images found.")
        return

    results = {}

    # 2. Process each folder
    for folder_path, image_list in folder_map.items():
        folder_name = os.path.basename(folder_path) or "Root"
        total_in_folder = len(image_list)
        print(f"\nProcessing folder: {folder_name} ({total_in_folder} images)")

        avg_hist = np.zeros((256, 1), dtype=np.float64)
        processed_count = 0

        # 3. Process the folder's images in batches
        for i in range(0, total_in_folder, batch_size):
            batch = image_list[i: i + batch_size]

            for path in batch:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # Calculate and normalize histogram
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                hist = hist / img.size

                # Incremental mean update
                processed_count += 1
                avg_hist += (hist - avg_hist) / processed_count

            print(f"  Progress: {processed_count}/{total_in_folder}", end='\r')

        results[folder_name] = avg_hist

    # 4. Final Comparison Plot
    plt.figure(figsize=(12, 8))
    for folder_name, hist in results.items():
        plt.plot(hist, label=folder_name, linewidth=2)
        plt.fill_between(range(256), hist.flatten(), alpha=0.1)

    plt.title(f'Comparison of Average Histograms (Batch Size: {batch_size})')
    plt.xlabel('Pixel Intensity (0-255)')
    plt.ylabel('Normalized Frequency')
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('folder_batch_histograms.png')
    plt.show()
    print("\nProcessing complete. Plot saved as 'folder_batch_histograms.png'.")


def show_random_peak_masks(base_path='.', peak_range=(170, 200), num_images=10):
    image_paths = []
    # Collect all image paths
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))

    if len(image_paths) < num_images:
        num_images = len(image_paths)

    # Select 10 random images
    selected_paths = random.sample(image_paths, num_images)

    # Setup plot: 10 rows, 2 columns (Original, Masked)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 3 * num_images))

    for i, path in enumerate(selected_paths):
        # Load image
        img = cv2.imread(path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create mask based on the peak intensity range
        # Only pixels between the specified intensities will be white (255)
        mask = cv2.inRange(gray, peak_range[0], peak_range[1])

        # Apply mask to original image to show only the peak parts
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        # Plotting
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Original: {os.path.basename(path)}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masked_img)
        axes[i, 1].set_title(f"Peak Intensity Mask ({peak_range[0]}-{peak_range[1]})")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('peak_intensity_samples.png')
    plt.show()


def process_hue_batches(base_path='.', batch_size=100):
    image_paths = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))

    total = len(image_paths)
    avg_hue_hist = np.zeros((180, 1), dtype=np.float64)
    processed_count = 0

    print(f"Starting Hue analysis for {total} images...")

    for i in range(0, total, batch_size):
        batch = image_paths[i: i + batch_size]
        for path in batch:
            img = cv2.imread(path)
            if img is None: continue

            # 1. Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)

            # 2. Filter: Only count pixels that are not too dark and not too gray
            # This prevents black/white areas from ruining your color data
            mask = cv2.bitwise_and(cv2.threshold(s, 50, 255, cv2.THRESH_BINARY)[1],
                                   cv2.threshold(v, 50, 255, cv2.THRESH_BINARY)[1])

            # 3. Calculate Hue Histogram (0-179)
            hist = cv2.calcHist([hsv], [0], mask, [180], [0, 180])

            # 4. Normalize and update incremental average
            valid_pixels = np.count_nonzero(mask)
            if valid_pixels > 0:
                hist = hist / valid_pixels
                processed_count += 1
                avg_hue_hist += (hist - avg_hue_hist) / processed_count

        print(f"Progress: {processed_count}/{total}", end='\r')

    # 5. Plotting with a Rainbow Background
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # Create rainbow background for easier reading
    for i in range(180):
        # Map 0-180 to 0-1.0 for the colormap
        color = plt.cm.hsv(i / 180.0)
        ax.axvspan(i, i + 1, color=color, alpha=0.15)

    plt.plot(avg_hue_hist, color='black', linewidth=2)
    plt.title(f'Average Hue Distribution ({processed_count} Images Analyzed)')
    plt.xlabel('Hue (Red -> Yellow -> Green -> Cyan -> Blue -> Violet -> Red)')
    plt.ylabel('Normalized Frequency')
    plt.xlim(0, 180)
    plt.grid(True, axis='y', alpha=0.3)

    plt.savefig('average_hue_distribution.png')
    plt.show()


def process_full_hsv_by_folder(base_path='.', batch_size=100):
    folder_map = {}
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    for root, _, files in os.walk(base_path):
        images = [os.path.join(root, f) for f in files if f.lower().endswith(valid_extensions)]
        if images:
            folder_map[root] = images

    if not folder_map:
        return "No images found."

    results = {}

    for folder_path, image_list in folder_map.items():
        folder_name = os.path.basename(folder_path) or "Root"
        h_avg, s_avg, v_avg = np.zeros((180, 1)), np.zeros((256, 1)), np.zeros((256, 1))
        count = 0

        for i in range(0, len(image_list), batch_size):
            for path in image_list[i: i + batch_size]:
                img = cv2.imread(path)
                if img is None: continue

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v_chan = cv2.split(hsv)

                count += 1
                # Update incremental means
                h_avg += (cv2.calcHist([hsv], [0], None, [180], [0, 180]) / img.size - h_avg) / count
                s_avg += (cv2.calcHist([hsv], [1], None, [256], [0, 256]) / img.size - s_avg) / count
                v_avg += (cv2.calcHist([hsv], [2], None, [256], [0, 256]) / img.size - v_avg) / count

        results[folder_name] = {'h': h_avg, 's': s_avg, 'v': v_avg}

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    for folder_name, data in results.items():
        # --- HUE (with Rainbow) ---
        ax_h = axes[0]
        if folder_name == list(results.keys())[0]:  # Only add rainbow background once
            for i in range(180):
                ax_h.axvspan(i, i + 1, color=plt.cm.hsv(i / 180.0), alpha=0.15)
        ax_h.plot(data['h'], label=folder_name, linewidth=2)
        ax_h.set_title('Average HUE (Color Identity)')
        ax_h.set_xlim(0, 180)

        # --- SATURATION (Gray to Vivid) ---
        ax_s = axes[1]
        ax_s.plot(data['s'], label=folder_name, linewidth=2)
        ax_s.set_title('Average SATURATION (Vividness)')
        ax_s.set_xlim(0, 255)

        # --- VALUE (Dark to Bright) ---
        ax_v = axes[2]
        ax_v.plot(data['v'], label=folder_name, linewidth=2)
        ax_v.set_title('Average VALUE (Brightness)')
        ax_v.set_xlim(0, 255)

    for ax in axes:
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.2)
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('full_hsv_analysis.png')
    plt.show()


def verify_all_peaks(base_path='.', num_samples=10):
    # Mapping the peaks from your charts to specific ranges
    peak_configs = [
        {'name': 'Overripe (Brown Hue 10-25)', 'type': 'hsv', 'low': (5, 40, 20), 'high': (25, 180, 120)},
        {'name': 'Ripe (Hue 20-35)', 'type': 'hsv', 'low': (15, 50, 50), 'high': (35, 255, 255)},
        {'name': 'Green (Hue 40-55)', 'type': 'hsv', 'low': (36, 40, 40), 'high': (60, 255, 255)},
        {'name': 'Lime (Hue ~75)', 'type': 'hsv', 'low': (70, 40, 40), 'high': (80, 255, 255)},
        {'name': 'Background (Hue ~115)', 'type': 'hsv', 'low': (100, 40, 40), 'high': (130, 255, 255)},
        {'name': 'Shadows (Val <10)', 'type': 'val', 'low': 0, 'high': 10},
        {'name': 'Highlights (Val 170-200)', 'type': 'val', 'low': 170, 'high': 200}
    ]

    # Collect paths
    image_paths = [os.path.join(r, f) for r, _, fs in os.walk(base_path)
                   for f in fs if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    selected = random.sample(image_paths, min(num_samples, len(image_paths)))
    cols = 1 + len(peak_configs)
    fig, axes = plt.subplots(len(selected), cols, figsize=(3 * cols, 4 * len(selected)))

    for i, path in enumerate(selected):
        img = cv2.imread(path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Column 0: Original
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].axis('off')
        if i == 0: axes[i, 0].set_title("Original Image")

        # Columns 1-6: The Peaks
        for j, cfg in enumerate(peak_configs, start=1):
            if cfg['type'] == 'hsv':
                mask = cv2.inRange(hsv, cfg['low'], cfg['high'])
            else:
                mask = cv2.inRange(gray, cfg['low'], cfg['high'])

            res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
            axes[i, j].imshow(res)
            axes[i, j].axis('off')
            if i == 0: axes[i, j].set_title(cfg['name'], fontsize=9)

    plt.tight_layout()
    plt.savefig('all_peaks_visual_check.png')
    plt.show()


def show_forced_peak_samples(base_path='.', target_hue=75, tolerance=3, num_to_find=10):
    # Define the specific Lime Green range
    lower_hsv = (target_hue - tolerance, 40, 40)
    upper_hsv = (target_hue + tolerance, 255, 255)

    found_images = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

    print(f"Scanning for images with significant content at Hue {target_hue}...")

    # 1. Search until we find 10 images that meet the criteria
    for root, _, files in os.walk(base_path):
        if len(found_images) >= num_to_find:
            break

        for file in files:
            if file.lower().endswith(valid_extensions):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                if img is None: continue

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

                # Check if the peak content covers at least 0.5% of the image
                if np.sum(mask > 0) > (img.size / 3 * 0.002):
                    found_images.append((path, img, mask))
                    print(f"  Found {len(found_images)}/{num_to_find}: {file}")

                if len(found_images) >= num_to_find:
                    break

    if not found_images:
        print("No images found with that specific hue peak.")
        return

    # 2. Display the results
    fig, axes = plt.subplots(len(found_images), 2, figsize=(10, 4 * len(found_images)))

    for i, (path, img, mask) in enumerate(found_images):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masked_res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Original: {os.path.basename(path)}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(masked_res)
        axes[i, 1].set_title(f"Content at Hue {target_hue} ±{tolerance}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('forced_peak_75_samples.png')
    plt.show()


if __name__ == "__main__":
    process_folders_in_batches(base_path="./Assets/bananadataset/dataset/data",batch_size=200)
    show_random_peak_masks(base_path="./Assets/bananadataset/dataset/data")
    process_full_hsv_by_folder(base_path="./Assets/bananadataset/dataset/data", batch_size=200)
    verify_all_peaks(base_path="./Assets/bananadataset/dataset/data", num_samples=20)
    show_forced_peak_samples(base_path="./Assets/bananadataset/dataset/data", target_hue=75, num_to_find=10)