# Nemotron-Labs-Diffusion 논문 리뷰

## 기본 정보

| 항목 | 내용 |
|---|---|
| 논문 | [Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding](https://arxiv.org/abs/2607.05722) |
| 저자 | Yonggan Fu et al. (26명) |
| 최초 공개일 | 2026-07-07 |
| 연구일 | 2026-08-06 |
| 정리·발표 | 김웅곤 (`kimwoonggon`) |
| 발표자료 | [PDF 열기](nemotron-labs-diffusion-paper-review.pdf) · 25쪽 |

## 한 문장 요약

AR과 masked discrete diffusion을 하나의 backbone에서 공동 학습해 AR, diffusion, self-speculation 세 가지 추론 모드를 상황에 따라 전환하는 tri-mode 언어 모델 연구다.

## 핵심 포인트

- 하나의 모델이 causal attention과 block-bidirectional attention을 전환해 세 추론 모드를 지원한다.
- 학습 목표는 `L_AR + 0.3 × L_diff`이며, 강한 AR prior가 diffusion 품질을 높이는 핵심 요소로 분석된다.
- Self-speculation에서는 같은 모델의 diffusion 모드가 여러 토큰을 draft하고 AR 모드가 검증한다.
- 8B 모델은 유사한 정확도에서 AR 대비 약 **6배 tokens per forward**를 보고한다.
- 실제 GB200 저동시성 환경에서는 최대 약 **4배 throughput**을 제시하지만, 동시성과 커널 구현에 따라 이득이 달라진다.

## 비판적으로 볼 점

- 6배 TPF와 4배 throughput은 같은 지표가 아니다.
- SOL(speed-of-light) 분석은 실제 배포 알고리즘이 아니라 oracle 상한이다.
- Sampler 학습 데이터의 도메인 편중과 긴 문맥에서의 일반화는 추가 검증이 필요하다.
- 고동시성에서는 기존 AR이 시스템 처리량 측면에서 다시 유리할 수 있다.

## 발표자료 구성

AR 병목 → 텍스트 diffusion → 공동 학습 → tri-mode inference → sampler와 self-speculation → 실제 성능 → 한계와 열린 문제 순으로 구성했다.

[← 전체 연구 목록으로 돌아가기](../../README.md)
