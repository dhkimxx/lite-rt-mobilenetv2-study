import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder
import numpy as np
import os
from typing import Tuple, Generator, List, Dict, Union


def get_calibration_dataset(
    data_type: str = "real",  # 'real' (CIFAR10) or 'random'
    n_samples: int = 100,
    batch_size: int = 1,
    image_size: int = 224,
    shuffle: bool = False,
) -> Generator[List[np.ndarray], None, None]:
    """
    Calibration을 위한 데이터 생성기 (Generator).
    """
    if data_type == "random":
        # Random Data Generator
        for _ in range(n_samples):
            # (B, C, H, W) -> numpy
            yield [
                np.random.randn(batch_size, 3, image_size, image_size).astype(
                    np.float32
                )
            ]
    else:
        # Real Data (ImageNet-V2 Matched-Frequency)
        # 1000 classes, 10 images per class (Total 10,000 images)
        # Matches ImageNet distribution better than CIFAR-10
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        root = "./data"
        dataset_url = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz"
        dataset_folder = os.path.join(root, "imagenetv2-matched-frequency-format-val")

        # Download & Extract if not exists
        if not os.path.exists(dataset_folder):
            print(f"[Setup] Downloading ImageNet-V2 from {dataset_url}...")
            try:
                download_and_extract_archive(
                    url=dataset_url,
                    download_root=root,
                    extract_root=root,
                    filename="imagenetv2-matched-frequency.tar.gz",
                    remove_finished=True,
                )
                print("[Setup] Download complete.")
            except Exception as e:
                print(f"[Warning] ImageNet-V2 download failed ({e}). Fallback to Random.")
                for _ in range(n_samples):
                    yield [
                        np.random.randn(batch_size, 3, image_size, image_size).astype(
                            np.float32
                        )
                    ]
                return

        # Load Dataset using ImageFolder
        try:
            dataset = ImageFolder(root=dataset_folder, transform=transform)
        except Exception as e:
            print(f"[Error] Failed to load ImageNet-V2 ({e}). Fallback to Random.")
            for _ in range(n_samples):
                yield [
                    np.random.randn(batch_size, 3, image_size, image_size).astype(
                        np.float32
                    )
                ]
            return

        # Subset for efficiency
        all_indices = list(range(len(dataset)))
        if shuffle:
            np.random.shuffle(all_indices)
        
        indices = all_indices[:n_samples]
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

        for images, _ in loader:
            yield [images.numpy()]


def compute_metrics(
    original_output: np.ndarray, quantized_output: np.ndarray
) -> Dict[str, float]:
    """
    MSE, SNR, Cosine Similarity를 계산합니다.
    """
    # Flatten
    idx = 0  # Batch index 0
    org = original_output.flatten()
    qtz = quantized_output.flatten()

    # 1. MSE
    mse = np.mean((org - qtz) ** 2)

    # 2. SNR (Signal-to-Noise Ratio)
    noise_power = mse
    signal_power = np.mean(org**2)

    if noise_power == 0:
        snr_db = float("inf")
    else:
        snr_db = 10 * np.log10(signal_power / noise_power)

    # 3. Cosine Similarity
    norm_org = np.linalg.norm(org)
    norm_qtz = np.linalg.norm(qtz)

    if norm_org == 0 or norm_qtz == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = np.dot(org, qtz) / (norm_org * norm_qtz)

    return {"MSE": float(mse), "SNR_dB": float(snr_db), "Cosine_Sim": float(cosine_sim)}
