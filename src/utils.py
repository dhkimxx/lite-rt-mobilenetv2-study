import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from typing import Tuple, Generator, List, Dict, Union


def get_calibration_dataset(
    data_type: str = "real",  # 'real' (CIFAR10) or 'random'
    n_samples: int = 100,
    batch_size: int = 1,
    image_size: int = 224,
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
        # Real Data (CIFAR10)
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Download usually creates a folder. We put it in a temporary location or cache.
        # Check if dataset exists, else download.
        root = "./data"
        os.makedirs(root, exist_ok=True)

        try:
            dataset = torchvision.datasets.CIFAR10(
                root=root, train=True, download=True, transform=transform
            )
        except Exception as e:
            print(
                f"Warning: CIFAR10 download failed ({e}). Falling back to Random data."
            )
            for _ in range(n_samples):
                yield [
                    np.random.randn(batch_size, 3, image_size, image_size).astype(
                        np.float32
                    )
                ]
            return

        # Subset for efficiency
        indices = list(range(n_samples))
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
