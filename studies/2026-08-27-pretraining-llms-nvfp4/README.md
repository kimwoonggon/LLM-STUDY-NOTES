# Pretraining Large Language Models with NVFP4 논문 리뷰

## 기본 정보

| 항목 | 내용 |
|---|---|
| 논문 | [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149) |
| 리뷰 기준 | arXiv:2509.25149v2 |
| 연구일 | 2026-08-27 |
| 정리·발표 | 김웅곤 (`kimwoonggon`) |
| 발표자료 | [PDF 열기](pretraining-llms-with-nvfp4-paper-review.pdf) · 31쪽 |

이번 주차는 **NVFP4 pretraining 논문 한 편**에 집중한다. [2026-08-14의 NVFP4·QAD 통합 리뷰](../2026-08-14-nvfp4-qad/README.md)에 이어, 숫자 형식의 기초와 Section 4의 Training Methodology를 심화했다. 발표 PDF는 발표자가 최종 정리한 파일을 내용 변경 없이 보관했다.

## 한 문장 요약

NVFP4로 linear GEMM 입력을 양자화하면서, 민감한 layer 보호·RHT·2D weight scaling·gradient stochastic rounding을 결합해 12B 모델의 10T-token 학습을 안정화한 방법을 다룬다.

## 논문이 푸는 문제

FP4는 표현할 수 있는 값이 적어 outlier, 반올림 오차, forward/backward의 양자화 표현 차이에 민감하다. 모든 linear layer에 NVFP4를 적용하고, 모든 tensor에 1×16 scaling과 round-to-nearest-even(RNE)을 사용하는 naive 설정은 조기에 발산한다. Section 4는 이 문제를 tensor와 연산의 역할에 맞춰 해결하는 과정이다.

## Training Methodology: 네 가지 문제와 해결책

| 문제 | 적용한 방법 | 이해할 때의 핵심 |
|---|---|---|
| 일부 layer의 큰 양자화 오차 | 민감한 linear layer를 BF16 등 고정밀로 유지 | 12B 주요 실험은 처음 2개와 마지막 8개 block의 linear layer를 BF16으로 보호한다. |
| Wgrad 입력의 block outlier | Wgrad 입력에 16×16 Random Hadamard Transform(RHT) | 큰 값을 여러 성분으로 분산한다. 양자화 전에는 같은 직교변환을 두 입력에 적용해 행렬곱을 보존할 수 있다. |
| transpose 뒤 weight 표현 불일치 | Weight에 16×16 2D scaling | 같은 tile의 scale을 사용해 forward와 backward의 quantized weight를 일관되게 만든다. |
| Gradient의 rounding bias | Gradient operand에 stochastic rounding(SR) | 가까운 두 값으로 확률적으로 반올림해 기대값의 편향을 줄인다. Weight와 activation은 RNE를 사용한다. |

짧게 기억하면 **W/A는 RNE, gradient는 SR, weight만 2D, Wgrad 입력만 RHT**다. Activation과 gradient는 1×16 scaling을 사용한다.

## NVFP4 형식에서 구분할 것

- NVFP4는 E2M1 FP4 값, block 단위 E4M3 scale, tensor 단위 FP32 scale을 결합한다.
- **Two-level scaling**은 tensor scale과 block scale을 함께 쓰는 방식이다.
- **2D weight scaling**은 16×16 weight tile에 scale을 공유하는 방식이다. Two-level scaling과 구분해야 한다.
- MXFP4와 비교하면 NVFP4는 기본 block이 더 작고 scale 표현이 더 촘촘하다.

## 실험 결과와 읽는 조건

- 주요 실험은 Nemotron-H 계열 12B hybrid Mamba-Transformer를 10T tokens로 학습하고, 같은 architecture의 FP8 baseline과 비교한다.
- Stable phase에서 relative loss gap은 1% 미만이지만, decay 종료 시점에는 약 1.5% 이상으로 벌어진다.
- 대부분의 downstream accuracy는 FP8에 가깝지만, HumanEval+와 MBPP+에서는 낮은 결과를 보인다. Table 2는 **BF16 평가**이며 NVFP4 inference 정확도 측정이 아니다.
- 같은 recipe를 적용한 8B 비교에서 MXFP4는 NVFP4와 같은 loss에 도달하는 데 36% 더 많은 token을 필요로 했다. 이 수치를 모든 모델의 일반적인 속도 차이로 해석하지 않는다.
- Ablation은 SR, Wgrad RHT, 2D weight scaling, 민감 layer 보호가 수렴에 기여함을 보여 준다.

## 발표자료 구성

아래 범위는 슬라이드에 적힌 번호가 아니라 **PDF의 실제 페이지 순서**를 기준으로 한다.

| PDF 페이지 | 내용 |
|---|---|
| 1–9 | 논문 핵심, 양자화와 scale, NVFP4·MXFP4 형식 |
| 10–14 | 12B·10T 실험 설정, precision 범위, loss·task accuracy |
| 15–24 | Naive FP4 실패, Fprop/Dgrad/Wgrad, 네 가지 해결책과 RHT 수식 |
| 25–28 | Ablation, MXFP4 비교, 한계와 핵심 결론 |
| 29–31 | 민감 layer 선택, SR·RHT 적용 위치, RHT 크기와 randomization 부록 |

## 한계와 비판적 질문

- 이 실험은 **mixed-precision training**이다. GEMM output, master weights, 누적 weight gradient, optimizer state와 여러 민감 경로는 BF16/FP32를 사용한다.
- Tensor Core 처리량 배수나 operand 메모리 절감만으로 end-to-end 학습 시간·에너지 개선을 단정할 수 없다.
- 더 큰 모델과 MoE로의 확장, 민감 layer의 자동 선택, attention·optimizer·communication까지 포함한 저정밀화는 추가 검증이 필요하다.

[← 전체 연구 목록으로 돌아가기](../../README.md)
