# NVFP4 Pretraining & QAD 논문 리뷰

## 기본 정보

| 항목 | 내용 |
|---|---|
| Paper A | [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149) |
| Paper A 저자·공개일 | NVIDIA et al. · 2025-09-29 |
| Paper B | [Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery](https://arxiv.org/abs/2601.20088) |
| Paper B 저자·공개일 | Meng Xin et al. · 2026-01-27 |
| 연구일 | 2026-08-14 |
| 정리·발표 | 김웅곤 (`kimwoonggon`) |
| 발표자료 | [PDF 열기](nvfp4-qad-paper-review.pdf) · 53쪽 |

## 한 문장 요약

NVFP4의 숫자 표현과 scale 원리부터 native FP4 pretraining의 수렴 recipe, 완성된 BF16 모델을 NVFP4 inference 모델로 바꿀 때의 QAD 정확도 복구까지 연결해서 정리한 두 논문 리뷰다.

## 두 논문이 푸는 문제

- **Paper A - NVFP4 Pretraining:** 모델을 처음부터 NVFP4 GEMM으로 학습하면서 장기 학습 수렴을 유지하는 방법을 다룬다.
- **Paper B - QAD:** post-training이 끝난 BF16 모델을 NVFP4로 양자화한 뒤 원래 모델의 행동 분포를 복구하는 방법을 다룬다.
- RHT, 2D scaling, stochastic rounding은 native pretraining recipe이고, frozen teacher와 forward KL은 QAD recipe다.

## 핵심 포인트

- NVFP4는 E2M1 FP4 값에 block 단위 E4M3 scale과 tensor 단위 FP32 scale을 결합한다.
- Native NVFP4 pretraining은 모든 연산을 FP4로 바꾸는 방식이 아니다. 큰 linear GEMM의 입력을 낮추고 output, 누산, master weights, optimizer state와 민감 경로는 고정밀로 유지한다.
- 단순 FP4 recipe의 수렴 문제를 selective BF16, WGRAD-only RHT, weight-only 2D scaling, gradient-only stochastic rounding의 조합으로 완화한다.
- Paper A는 12B 모델을 10T token 동안 학습하고 FP8 baseline에 가까운 loss와 downstream 결과를 보고한다.
- QAD는 frozen BF16 teacher와 simulated-NVFP4 student의 next-token 분포 사이에서 `D_KL(p_teacher || p_student)`를 최소화한다.
- QAD는 특히 SFT, RL, merge가 포함된 multi-stage post-training 모델에서 일반 QAT보다 원래 행동 분포를 보존하는 데 유리한 결과를 보였다.

## 비판적으로 볼 점

- Paper A는 알고리즘적 수렴 가능성을 보이지만 end-to-end training wall-clock speedup을 직접 보고하지 않는다.
- Native NVFP4 training은 mixed precision이며 모델의 모든 tensor와 연산이 4비트인 것은 아니다.
- QAD의 near-BF16 recovery는 모든 모델과 benchmark에서 완전 복원을 보장하지 않는다.
- QAD는 label-free일 수 있지만 input-free는 아니며, BF16 teacher forward에 따른 compute와 memory 비용이 필요하다.
- QAD는 RL rollout 중 sampler-learner mismatch를 직접 해결하는 방법이 아니다.

## 발표자료 구성

Quantization 기초 → INT4와 FP4 → NVFP4의 local/global scale → native FP4 training의 실패 원인 → RHT·2D scaling·stochastic rounding·selective BF16 → PTQ·QAT·QAD → fake quantization과 STE → forward KL → QAD 실험 결과 → Nemotron production stack → 예상 질문 10개 순으로 구성했다.

[← 전체 연구 목록으로 돌아가기](../../README.md)
