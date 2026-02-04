# MobileNetV2 양자화 및 온디바이스 성능 분석

## 1. 서론 (Introduction)

본 연구의 목적은 MobileNetV2 모델을 모바일 장치(Edge Device)에 효율적으로 배포하기 위한 최적의 양자화(Quantization) 전략을 도출하는 것이다. 딥러닝 모델의 경량화는 제한된 컴퓨팅 자원과 전력 소모를 가진 모바일 환경에서 필수적이다. 본 프로젝트에서는 TensorFLow Lite(TFLite) 프레임워크를 기반으로 다양한 양자화 기법(Post-Training Quantization)을 적용하고, 이를 실제 안드로이드 디바이스(Samsung Galaxy A7, arm64-v8a)에서 벤치마킹하여 정확도(Accuracy)와 추론 속도(Latency) 간의 트레이드오프를 분석하였다.

---

## 2. 실험 환경 및 방법 (Methodology)

### 2.1. 하드웨어 및 소프트웨어 환경

- **Host Environment**: Linux (x86_64), Python 3.10
- **Target Device**: Samsung Galaxy A7 (SM-A750N), CPU: Octa-core (Cortex-A73/A53), GPU: Mali-G71 MP2
- **Frameworks**: `torch`, `ai-edge-quantizer`, `tensorflow-lite`
- **Benchmark Tool**: TensorFlow Lite Benchmark Binary (Nightly, Android aarch64)

### 2.2. 모델 및 데이터셋

- **Model**: MobileNetV2 (Pre-trained on ImageNet)
- **Calibration Dataset**: ImageNet-V2 (Matched Frequency, 1000 classes, 10k images)
- **Metrics**:
  - **MSE (Mean Squared Error)**: 원본 FP32 모델과의 출력 텐서 차이
  - **SNR (Signal-to-Noise Ratio)**: 양자화 노이즈 대비 신호 비율 (dB)
  - **Latency**: 온디바이스 평균 추론 속도 (ms)
  - **Model Size**: TFLite 파일 크기 (MB)

### 2.3. 실험 설계 (Experimental Matrix)

총 6가지 대표 양자화 전략과 3가지 런타임 환경을 조합하여 평가하였다.

| 실험 ID  | 양자화 방식 (Method)   | 설명 (Description)                                                   |
| :------- | :--------------------- | :------------------------------------------------------------------- |
| **C0**   | **Baseline (FP32)**    | 양자화 미적용 (Float32). 성능 기준점.                                |
| **C1**   | **Weight-Only**        | 가중치(Weight)만 INT8로 변환. 연산은 FP32로 수행.                    |
| **C2**   | **Dynamic Range**      | 가중치 INT8 고정, 활성화(Activation)는 실행 시 동적 양자화.          |
| **C3-C** | **Static (Full INT8)** | 가중치/활성화 모두 INT8 고정. Calibration 데이터 필요 (100 samples). |
| **C4**   | **Full Integer**       | 입력/출력 텐서까지 모두 INT8 (TPU/NPU 호환).                         |
| **C5**   | **Per-Tensor**         | 채널 단위가 아닌 텐서 단위 양자화 (비교군).                          |

---

## 3. 실험 결과 (Results)

### 3.1. 정확도 및 모델 크기 (Accuracy & Size)

| ID       | SNR (dB)  | Model Size (MB) | 비고                     |
| :------- | :-------- | :-------------- | :----------------------- |
| **C0**   | ∞         | 13.4            | Baseline                 |
| **C1**   | 24.33     | 3.74            | **72% Size Reduction**   |
| **C2**   | 23.90     | 3.71            | High Accuracy            |
| **C3-C** | 22.77     | 3.92            | Stable INT8              |
| **C4**   | 22.77     | 3.92            | NPU Compatible           |
| **C5**   | **15.50** | 3.51            | **Accuracy Degradation** |

- **분석**:
  - **Per-Channel (C3, C4) vs Per-Tensor (C5)**: Depthwise Convolution이 많은 MobileNetV2 특성상, Per-Tensor 양자화(C5)는 SNR이 급격히 하락함(15.5dB). Per-Channel 방식이 필수적임.
  - **용량 절감**: 모든 양자화 모델이 FP32 대비 약 1/4 수준으로 용량이 감소함.

### 3.2. 온디바이스 추론 속도 (On-Device Latency)

Samsung A7 디바이스에서 측정한 평균 추론 속도 (단위: ms).

| Model             | CPU (1 Thread) | CPU (4 Threads) | GPU Delegate   |
| :---------------- | :------------- | :-------------- | :------------- |
| **C0 (FP32)**     | 70.9 ms        | 32.4 ms         | **16.6 ms**    |
| **C1 (W-Only)**   | 70.7 ms        | 31.6 ms         | _N/A (Failed)_ |
| **C2 (Dynamic)**  | 69.2 ms        | 94.6 ms\*       | 16.3 ms        |
| **C3-C (Static)** | 43.2 ms        | **20.5 ms**     | 19.0 ms        |
| **C4 (Full Int)** | 43.1 ms        | 22.0 ms         | 19.1 ms        |

> _Note: C2 CPU-4T의 느린 속도(94.6ms)는 동적 양자화 오버헤드와 멀티스레딩 컨텍스트 스위칭 비용이 복합적으로 작용한 것으로 추정됨._

- **분석**:
  - **CPU 가속**: Static Quantization(C3, C4) 적용 시 FP32(C0) 대비 **약 1.5배 (32ms -> 20ms)** 속도 향상.
  - **GPU 가속**: GPU 환경에서는 오히려 **FP32(C0)가 가장 빠름(16.6ms)**. INT8 모델(19ms)은 GPU 내부에서 Dequantization 오버헤드가 발생하거나, 해당 GPU(Mali-G71)가 INT8 가속을 네이티브로 지원하지 않아 FP16으로 변환하여 처리하는 것으로 분석됨.
  - **Weight-Only(C1)**: GPU Delegate에서 지원하지 않아 실행 실패.

---

## 4. 결론 (Conclusion)

본 연구를 통해 MobileNetV2 모델의 모바일 최적화 전략을 다음과 같이 제안한다.

1.  **CPU 환경 (일반적인 앱 배포)**:
    - **추천**: **Static Quantization (C3/C4)**
    - **이유**: FP32 대비 **1.5배 빠른 속도**와 **4배 작은 모델 크기**를 제공하며, SNR 손실(약 22dB 유지)이 허용 가능한 수준이다.

2.  **GPU 환경 (고성능 필요 시)**:
    - **추천**: **Baseline FP32 (C0)**
    - **이유**: 해당 테스트 기기(Mali GPU)에서는 FP32/FP16 연산이 가장 최적화되어 있다. 양자화 모델은 오히려 오버헤드를 유발한다.

3.  **데이터 의존성**:
    - **Per-Tensor 양자화 지양**: 정확도 손실이 매우 크므로 반드시 **Per-Channel** 방식을 사용해야 한다.
    - **Calibration**: Random Data로도 수행 가능하나, 실서비스 수준의 정확도를 위해서는 실제 데이터 분포(Representitive Dataset)를 사용하는 것을 권장한다.

## 5. 사용법 (Usage)

### 5.1 실험 재현

```bash
# 1. 의존성 설치
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. 모델 변환 및 정량적 평가 (PC)
uv run src/run_experiments.py

# 3. 온디바이스 벤치마크 (Android 기기 연결 필요)
# benchmark_model 바이너리가 루트 디렉토리에 있어야 함
uv run src/benchmark_adb.py
```

### 5.2 결과 리포트

- [`notebooks/Quantization_Report.ipynb`](notebooks/Quantization_Report.ipynb): 상세 데이터 시각화(MSE, SNR, 온디바이스 Latency 비교) 및 분석 리포트.
- `results/android_benchmark_results.csv`: 온디바이스 벤치마크 로우 데이터.
