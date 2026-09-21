# LLM Study Notes

LLM 논문을 읽고 **핵심 아이디어, 시스템 구조, 수식, 실험 결과, 한계**를 한국어로 정리한 연구 아카이브입니다.

![Studies](https://img.shields.io/badge/studies-7-2563eb?style=flat-square)
![Language](https://img.shields.io/badge/language-Korean-0f766e?style=flat-square)
![Format](https://img.shields.io/badge/slides-PDF%20%7C%20PPTX%20%7C%20HTML-dc2626?style=flat-square)
![Last update](https://img.shields.io/badge/last_update-2026--09--21-475569?style=flat-square)

> 정리자와 발표자는 각 연구 항목에 별도로 기록합니다.<br>
> 발표자료는 GitHub에서 바로 열어 보거나 PDF로 내려받을 수 있습니다.

## 연구 목록

최신 연구가 위에 오도록 정리했습니다.

| 연구일 | 논문 | 핵심 주제 | 자료 | 발&#8288;표&#8288;자 |
|---|---|---|---|---|
| 2026-09-11 | [QeRL: Beyond Efficiency – Quantization-enhanced Reinforcement Learning for LLMs](https://arxiv.org/abs/2510.11696v1) | NVFP4·LoRA 강화학습과 AQN의 탐색 조절, 모델별 정확도·효율 비교 | [연구 요약](studies/2026-09-11-qerl/README.md) · [발표 PDF](studies/2026-09-11-qerl/qerl-paper-review.pdf) · [PPTX](studies/2026-09-11-qerl/qerl-paper-review.pptx) · [HTML](studies/2026-09-11-qerl/qerl-paper-review.html) | [김&#8288;웅&#8288;곤](https://github.com/kimwoonggon) |
| 2026-09-06 | [Quantized Reasoning Models Think They Need to Think Longer, but They Do Not](https://arxiv.org/abs/2606.00206) | 강한 PTQ의 overthinking 오류와 추론 시점 logit penalty | [연구 요약](studies/2026-09-06-quantized-reasoning-overthinking/README.md) · [발표 PDF](studies/2026-09-06-quantized-reasoning-overthinking/quantized-reasoning-models-paper-review.pdf) · [PPTX](studies/2026-09-06-quantized-reasoning-overthinking/quantized-reasoning-models-paper-review.pptx) | [김&#8288;웅&#8288;곤](https://github.com/kimwoonggon) |
| 2026-08-27 | [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149) | 12B·10T NVFP4 학습과 Training Methodology의 네 가지 안정화 장치 | [연구 요약](studies/2026-08-27-pretraining-llms-nvfp4/README.md) · [발표 PDF](studies/2026-08-27-pretraining-llms-nvfp4/pretraining-llms-with-nvfp4-paper-review.pdf) | [김&#8288;웅&#8288;곤](https://github.com/kimwoonggon) |
| 2026-08-14 | [NVFP4 Pretraining](https://arxiv.org/abs/2509.25149) + [QAD](https://arxiv.org/abs/2601.20088) | Native FP4 학습 수렴과 post-hoc NVFP4 정확도 복구 | [연구 요약](studies/2026-08-14-nvfp4-qad/README.md) · [발표 PDF](studies/2026-08-14-nvfp4-qad/nvfp4-qad-paper-review.pdf) | [김&#8288;웅&#8288;곤](https://github.com/kimwoonggon) |
| 2026-08-06 | [Nemotron-Labs-Diffusion](https://arxiv.org/abs/2607.05722) | AR·diffusion·self-speculation을 하나의 모델로 통합 | [연구 요약](studies/2026-08-06-nemotron-labs-diffusion/README.md) · [발표 PDF](studies/2026-08-06-nemotron-labs-diffusion/nemotron-labs-diffusion-paper-review.pdf) | [김&#8288;웅&#8288;곤](https://github.com/kimwoonggon) |
| 2026-07-30 | [Accelerating RL Post-Training Rollouts via System-Integrated Speculative Decoding](https://arxiv.org/abs/2604.26779) | RL rollout에 speculative decoding을 시스템 단위로 통합 | [연구 요약](studies/2026-07-30-rl-post-training-speculative-decoding/README.md) · [발표 PDF](studies/2026-07-30-rl-post-training-speculative-decoding/rl-post-training-speculative-decoding-paper-review.pdf) · [상세 노트](studies/2026-07-30-rl-post-training-speculative-decoding/study-notes-ko.md) | [김&#8288;웅&#8288;곤](https://github.com/kimwoonggon) |
| 2026-07-23 | [DeepSeek-V4](https://arxiv.org/abs/2606.19348) | 1M 컨텍스트를 위한 압축 attention과 post-training 설계 | [연구 요약](studies/2026-07-23-deepseek-v4/README.md) · [발표 PDF](studies/2026-07-23-deepseek-v4/deepseek-v4-paper-review.pdf) | [김&#8288;웅&#8288;곤](https://github.com/kimwoonggon) |

## 빠르게 보는 법

1. 위 표에서 **연구 요약**을 열면 논문 정보와 핵심 결론을 먼저 볼 수 있습니다.
2. **발표 PDF**는 GitHub PDF 뷰어에서 바로 읽을 수 있습니다.
3. 더 깊은 내용이 있는 경우 **상세 노트**에 발표 준비 과정과 예상 질문을 함께 기록합니다.
4. 원문 확인이 필요하면 논문 제목을 눌러 공식 arXiv 페이지로 이동합니다.
5. **PPTX**가 함께 등록된 연구는 편집 가능한 발표 원본도 내려받을 수 있습니다.
6. **HTML** 발표자료는 파일을 내려받아 브라우저에서 열면 조작형 그래프와 발표자 노트를 사용할 수 있습니다. GitHub에서는 HTML 소스가 표시됩니다.

## 폴더 구조

```text
studies/
├── YYYY-MM-DD-paper-slug/
│   ├── README.md                 # 논문 정보와 핵심 연구 요약
│   ├── *-paper-review.pdf        # 한국어 발표자료
│   ├── *-paper-review.pptx       # 편집 가능한 발표 원본이 있는 경우
│   ├── *-paper-review.html       # 브라우저용 발표자료가 있는 경우
│   └── study-notes-ko.md         # 상세 학습 노트가 있는 경우
```

## 기록 원칙

- **연구일**은 발표자료를 완성한 날짜를 기준으로 합니다. 날짜를 확인할 수 없는 경우 자료 등록일을 사용하고 해당 연구에 명시합니다.
- 발표자와 공동 정리자는 연구별 **발표자** 항목에 함께 기록합니다.
- 논문의 공개일, 저자, 제목은 공식 arXiv 정보를 기준으로 기록합니다.
- 수치와 결론은 실제 측정값과 시뮬레이션·전망치를 구분합니다.
- 요약뿐 아니라 재현성, 비교 조건, 한계와 비판적 질문을 함께 남깁니다.

## 저작권 안내

논문 원문과 논문 속 도표의 권리는 각 저자 및 배포처에 있습니다. 이 저장소에는 개인이 작성한 연구 요약과 발표자료만 보관하며, 논문 원문은 공식 링크로 연결합니다.
