# DeepSeek-V4 논문 리뷰

## 기본 정보

| 항목 | 내용 |
|---|---|
| 논문 | [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) |
| 저자 | DeepSeek-AI et al. |
| 최초 공개일 | 2026-04-26 |
| 연구일 | 2026-07-23 |
| 정리·발표 | 김웅곤 (`kimwoonggon`) |
| 발표자료 | [PDF 열기](deepseek-v4-paper-review.pdf) · 20쪽 |

## 한 문장 요약

DeepSeek-V4는 1M 토큰 컨텍스트를 단순히 지원하는 데 그치지 않고, 압축 attention·MoE·학습 안정화·post-training을 함께 설계해 실제 비용 구조를 낮추려는 시스템 보고서다.

## 핵심 포인트

- **CSA + HCA**: 중간·강한 압축과 희소·밀집 조회를 조합하고 최근 토큰은 sliding window로 보존한다.
- **효율 주장**: 1M 컨텍스트에서 V4-Pro는 V3.2 대비 single-token inference FLOPs 27%, KV cache 10%를 제시한다.
- **학습 안정화**: mHC와 Muon optimizer를 대형 MoE 학습에 결합한다.
- **Post-training 분업**: GRPO로 도메인 전문가를 강화하고, OPD로 여러 전문가의 능력을 하나의 모델에 통합한다.
- **해석 주의**: FLOPs 감소가 같은 비율의 실제 latency 감소를 뜻하지는 않는다.

## 비판적으로 볼 점

- CSA, HCA, mHC, Muon, 데이터 구성 각각의 인과를 분리하는 ablation이 제한적이다.
- 전체 학습 compute, 에너지, 비용과 32T 이상 학습 데이터의 세부 구성이 충분히 공개되지 않았다.
- 1M context 지원과 1M 구간에서의 손실 없는 품질은 서로 다른 주장이다.

## 발표자료 구성

병목 정의 → CSA/HCA 구조 → MoE 학습 안정화 → GRPO/OPD → 성능 비교 → 한계와 예상 질문 순으로 구성했다.

[← 전체 연구 목록으로 돌아가기](../../README.md)
