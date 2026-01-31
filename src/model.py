import torch
import torchvision
from torchvision.models import MobileNet_V2_Weights


def get_model(pretrained: bool = True):
    """
    MobileNetV2 모델을 로드합니다.

    Args:
        pretrained (bool): Pre-trained 가중치(ImageNet) 사용 여부. 기본값은 True.

    Returns:
        torch.nn.Module: 평가 모드(eval)로 설정된 MobileNetV2 모델.
    """
    # 사전 학습된 가중치 로드 (IMAGENET1K_V1)
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.mobilenet_v2(weights=weights)

    # 양자화 및 추론을 위해 평가 모드로 설정
    model.eval()
    return model


def get_representative_dataset(n_samples: int = 100):
    """
    양자화 Calibration(보정)을 위한 Representative Dataset을 생성합니다.
    실제 데이터셋 대신 정규 분포를 따르는 Random Tensor를 사용하여 구조적 양자화 테스트를 수행합니다.

    Args:
        n_samples (int): 생성할 샘플 데이터의 개수 (기본값: 100)

    Yields:
        list[torch.Tensor]: 모델 입력 형태(1, 3, 224, 224)에 맞는 텐서 리스트.
                           tflite 변환기는 일반적으로 입력 인자 리스트를 기대합니다.
    """
    for _ in range(n_samples):
        # MobileNetV2 입력 규격: (Batch Size: 1, Channel: 3, Height: 224, Width: 224)
        # 실제 환경에서는 전처리된 이미지(Normalization 적용)를 사용해야 정확도가 보장되지만,
        # 본 실험은 '양자화 과정의 동작 검증'이 주 목적이므로 랜덤 데이터를 사용합니다.
        data = torch.randn(1, 3, 224, 224)
        yield [data]
