import csv
from collections import Counter
import numpy as np

def extract_client_metadata_from_loader(
    cid: int,
    labels: np.ndarray,
    is_clean: bool,
    num_classes: int = 10,
):
    """Return a dict with all client properties."""
    class_counts = Counter(labels.tolist())

    metadata = {
        "cid": cid,
        "num_samples": int(len(labels)),
        "is_clean": int(is_clean),
    }

    for c in range(num_classes):
        metadata[f"class_{c}_samples"] = int(class_counts.get(c, 0))

    return metadata


def save_clients_metadata_csv(metadata_list, output_path):
    """Save all client metadata to CSV."""
    fieldnames = metadata_list[0].keys()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metadata_list:
            writer.writerow(row)