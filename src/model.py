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