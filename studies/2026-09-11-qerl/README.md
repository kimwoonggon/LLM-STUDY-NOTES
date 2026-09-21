# QeRL 논문 리뷰

## 기본 정보

| 항목 | 내용 |
|---|---|
| 논문 | [QeRL: Beyond Efficiency – Quantization-enhanced Reinforcement Learning for LLMs](https://arxiv.org/abs/2510.11696v1) |
| 저자 | Wei Huang 외 13명 |
| 최초 제출일 | 2025-10-13 |
| 리뷰 기준 | arXiv:2510.11696v1 |
| 연구일 | 2026-09-11 (34장 발표자료 완성일) |
| 자료 등록일 | 2026-09-21 |
| 정리·발표 | 김웅곤 (`kimwoonggon`) |
| 발표자료 | [PDF 열기](qerl-paper-review.pdf) · 34쪽 · [PPTX 다운로드](qerl-paper-review.pptx) · 34장 · [HTML 발표자료](qerl-paper-review.html) |
| 공식 구현 | [NVlabs/QeRL](https://github.com/NVlabs/QeRL) |

AQN과 LoRA의 수식 비교를 보강한 34장 자료를 보관했다. PDF는 같은 PPTX에서 내보낸 버전이다. PPTX의 본문·표와 재작성한 그래프는 편집할 수 있으며, 수식과 논문 원본 그림은 배치를 유지하도록 이미지로 넣었다. 각 슬라이드의 발표자 노트에는 설명과 근거 위치를 기록했다.

HTML은 파일을 내려받아 브라우저에서 연다. 인터넷 연결 없이 발표할 수 있으며, 모델 선택과 엔트로피·잡음 감쇠·생성 비중을 조작할 수 있다. PPTX와 PDF에는 해당 요소를 대표적인 정적 상태로 담았다. 논문 원문은 저장소에 복제하지 않고 공식 링크로 연결한다.

## 한 문장 요약

QeRL은 NVFP4로 기본 가중치를 압축하고 LoRA로 강화학습하며, 양자화와 추가 잡음이 풀이 탐색을 도울 수 있음을 보여 준다. 다만 AQN의 정확도 이득은 모델과 평가 과제에 따라 달라진다.

## 논문이 푸는 문제

LLM 강화학습은 답안을 여러 개 생성하는 롤아웃과 정책 갱신에 큰 메모리·시간 비용이 든다. LoRA는 학습할 파라미터 수를 줄이지만 큰 기본 가중치를 계속 읽어야 한다. QeRL은 기본 가중치의 저장 형식과 생성 커널을 바꾸어 비용을 줄이면서, 저정밀 표현의 오차가 학습 중 탐색에 미치는 영향도 분석한다.

**QeRL은 새 기본 모델의 이름이 아니라 학습 구성이다.** 이 자료에서는 NVFP4 LoRA를 QeRL로, 추가 잡음 조절을 켠 설정을 QeRL + AQN으로 구분한다.

## 실험 설정

| 구분 | 내용 |
|---|---|
| 기본 모델 | Qwen2.5-3B·7B·14B·32B-Instruct |
| GSM8K 실험 | 3B·7B를 GSM8K로 GRPO 학습한 뒤 GSM8K 정답률 평가 |
| BigMath 실험 | 7B·14B·32B를 BigMath로 DAPO 학습한 뒤 MATH 500, AIME 24·25, AMC 23 평가 |
| 비교 방식 | BF16 Full, BF16 LoRA, NF4 QLoRA, MXFP4 LoRA, NVFP4 LoRA, NVFP4 LoRA + AQN. 표마다 실제 보고된 방식을 구분한다. |
| LoRA rank | 주요 비교는 32. 별도 실험에서 16·32·64·128 비교 |
| 정확도 실험 하드웨어 | H100 80GB 8장 |
| 효율 실험 하드웨어 | H100 80GB 1장. 모델·배치·gradient checkpointing 조건을 별도 표시 |

근거: [논문 §4.1, Table 4, Appendix F](https://arxiv.org/html/2510.11696v1). 32B의 단일 GPU 학습 가능성 실험과 최종 정확도 실험의 GPU 수를 혼동하지 않는다.

## LoRA와 AQN은 각각 무엇을 바꾸나

### LoRA: 보상으로 학습한 작은 보정값을 남긴다

입력을 열벡터로 쓰면 한 선형층은 다음과 같이 표현할 수 있다.

$$
y=(W+sBA)x,\qquad s=\frac{\alpha}{r}
$$

$W\in\mathbb{R}^{m\times d}$는 고정한 기본 가중치다. $A\in\mathbb{R}^{r\times d}$와 $B\in\mathbb{R}^{m\times r}$만 학습하므로, 이 층의 학습 파라미터 수는 $md$에서 $r(d+m)$로 줄어든다. $sBA$는 기울기 업데이트가 누적되고 저장되는 보정값이다. QeRL에서는 기본 가중치를 NVFP4로 저장하며, 복원한 값을 $\widehat W$로 표기한다.

근거: [LoRA §4.1](https://arxiv.org/html/2106.09685v2#S4.SS1), [QeRL §2, 식 (2)](https://arxiv.org/html/2510.11696v1#S2). QeRL 식 (2)에서 생략한 LoRA 배율 $\alpha/r$을 여기서는 명시했다.

### AQN: 정규화 배율을 잠시 흔들어 다른 풀이를 탐색한다

AQN(Adaptive Quantization Noise)은 RMSNorm의 채널별 배율에 작은 무작위 잡음을 더한다. 정규화한 입력을 $u(x)$, 원래 배율을 $w$라고 쓰면 다음과 같다.

$$
h=w\odot u(x),\qquad h_k=(w+z_k)\odot u(x),\qquad z_{k,j}\sim\mathcal N(0,\sigma_k^2)
$$

잡음은 매 순전파마다 다시 뽑는다. $\sigma_k$는 표준편차이며 학습 단계에 따라 정한 일정으로 줄인다. 잡음 자체를 보상으로 학습하는 것은 아니고, 보상에 따라 학습되는 것은 LoRA의 $A,B$다. 여기서 adaptive는 보상·엔트로피를 측정해 자동으로 세기를 결정하는 피드백 제어라는 뜻이 아니다.

발표자료의 감쇠 예시는 처음 추가 잡음을 끄는 구간과 감쇠 구간을 구분하고, 감쇠 구간을 아래처럼 표현했다.

$$
\sigma_k=\sigma_{\mathrm{start}}\left(\frac{\sigma_{\mathrm{end}}}{\sigma_{\mathrm{start}}}\right)^{\frac{k-1}{K-1}},\qquad k=1,\ldots,K
$$

시작값 0.01과 끝값 0.0005는 방법 설명과 Table 4를 따랐다. $K=10$인 그래프는 식을 이해하기 위한 예시이며 실제 학습 로그가 아니다. 논문 §4.1에는 시작값 0.05가 적혀 있고, 식 (8)과 Algorithm 1의 단계 번호에도 차이가 있어 발표자 노트에 이를 밝혔다.

RMSNorm 바로 다음 선형층을 설명할 때, $w_j\ne0$인 채널에서는 다음처럼 다시 쓸 수 있다.

$$
D_k=\operatorname{diag}(1+z_k\oslash w),\qquad h_k=D_kh,\qquad y=(\widehat W+sBA)D_kh
$$

$\oslash$는 원소별 나눗셈이다. 이 식은 바뀐 정규화 출력을 기본 분기와 LoRA 분기가 함께 받는 경우를 나타낸 설명용 결합식이며, 바이어스와 dropout은 생략했다. 실제로 큰 대각행렬을 만들거나 NVFP4 가중치에 고정밀 잡음을 직접 더하는 것은 아니다. RMSNorm의 배율을 바꾸므로 저정밀 가중치의 연산 경로를 유지할 수 있다.

근거: [QeRL §3.3, 식 (8), (10)–(12), Figure 6, Appendix G](https://arxiv.org/html/2510.11696v1). 결합식은 논문의 행벡터 표기를 열벡터로 통일하여 설명한 것이다.

## 실험 결과와 읽는 조건

### GSM8K: 같은 크기의 Qwen2.5-Instruct끼리 비교

GSM8K로 GRPO 학습한 뒤 평가한 정답률(%). Full은 전체 파라미터 학습이다.

| 학습 방식 | 3B | 7B |
|---|---:|---:|
| BF16 Full | 84.4 | 91.2 |
| BF16 LoRA | 76.1 | 88.1 |
| NF4 QLoRA | 76.1 | 85.0 |
| MXFP4 LoRA | 73.4 | 86.4 |
| QeRL (NVFP4 LoRA) | 83.3 | 88.5 |
| QeRL + AQN | 83.7 | 90.8 |

QeRL + AQN은 BF16 LoRA보다 3B에서 7.6%p, 7B에서 2.7%p 높다. NF4 QLoRA 대비 차이는 각각 7.6%p, 5.8%p다. 7B의 Full 91.2%와 QeRL + AQN 90.8%는 가까운 결과지만 같은 수치는 아니다. 근거: [Table 1](https://arxiv.org/html/2510.11696v1).

### BigMath: Qwen2.5-7B-Instruct를 DAPO로 학습한 결과

BigMath 난도 3–5로 학습한 뒤 평가한 Pass@1(%). 평균은 논문 보고값을 그대로 옮겼다.

| 학습 방식 | MATH 500 | AIME 24 | AIME 25 | AMC 23 | 평균 |
|---|---:|---:|---:|---:|---:|
| BF16 LoRA | 77.0 | 13.3 | 10.0 | 42.5 | 35.7 |
| QeRL (NVFP4 LoRA) | 76.8 | 13.7 | 10.0 | 47.5 | 37.0 |
| QeRL + AQN | 77.4 | 15.5 | 10.0 | 42.5 | 36.4 |
| BF16 Full | 77.4 | 16.7 | 10.0 | 45.0 | 37.3 |

QeRL + AQN의 MATH 500은 Full과 같은 77.4%다. 그러나 AQN 추가 후 AMC 23은 47.5%에서 42.5%로 낮아지고, 평균도 37.0%에서 36.4%로 낮아진다. **특정 과제의 동률을 모든 과제의 동등한 성능으로 확대하지 않는다.** 근거: [Table 2의 7B 결과](https://arxiv.org/html/2510.11696v1).

### 모델 크기에 따른 AQN의 효과

Qwen2.5-Instruct를 BigMath + DAPO로 학습한 뒤, 위 네 수학 벤치마크의 평균 정확도(%)를 비교한다.

| 학습 방식 | 7B | 14B | 32B |
|---|---:|---:|---:|
| BF16 LoRA | 35.7 | 40.2 | 42.2 |
| QeRL | 37.0 | 40.5 | 41.4 |
| QeRL + AQN | 36.4 | 42.0 | 45.6 |
| BF16 Full | 37.3 | 43.3 | 46.2 |

AQN 유무만 비교하면 7B −0.6%p, 14B +1.5%p, 32B +4.2%p다. 양자화 직후와 강화학습 후의 차이에는 강화학습 자체의 효과도 포함되므로, 그 증가분 전체를 AQN의 효과로 해석하지 않는다. 근거: [Table 2](https://arxiv.org/html/2510.11696v1).

### 효율: 저장 크기, 롤아웃, 전체 학습을 구분

- Qwen2.5-7B·14B·32B-Instruct의 저장된 모델 크기는 BF16 LoRA 대비 약 61–67% 줄었다. 이는 학습 중 최고 GPU 메모리 사용량과 다른 지표다.
- H100 80GB 1장의 7B·14B GRPO 실험에서 QeRL의 전체 학습 속도는 BF16 LoRA 대비 1.2–1.5배였다. 모델과 배치에 따라 달라진다.
- 배치 8의 32B 롤아웃 처리량은 344.3에서 688.2 tokens/s로 약 2배가 됐다. 이 수치를 전체 학습 배속으로 바꾸어 말하지 않는다.
- 32B는 gradient checkpointing을 켜면 H100 한 장에서 QeRL 학습이 가능했다. 같은 조건의 BF16 LoRA는 메모리 부족(OOM)이므로 두 방식의 전체 학습 배속을 계산할 수 없다.

근거: [Table 3, Tables 5–8, Appendix I](https://arxiv.org/html/2510.11696v1).

## 왜 좋아졌을까: 관측과 해석을 나누어 읽기

| 관측한 결과 | 설명 가능한 원인 | 해석의 범위 |
|---|---|---|
| 모델 저장 크기 감소 | 가중치당 저장 비트 수 감소 | 저장 크기로 직접 확인. 총 학습 메모리는 별도 측정 필요 |
| 롤아웃 속도 개선 | 가중치 전송량 감소와 NVFP4 커널의 처리 방식 | 같은 4비트라도 NF4와 NVFP4의 속도는 다를 수 있음 |
| 높은 정책 엔트로피와 빠른 보상 증가 | 다른 토큰·풀이를 시도해 유효한 정답 경로를 발견할 가능성 | 다양한 오답만 늘면 정확도 개선으로 이어지지 않음 |
| AQN의 선택적인 성능 향상 | 탐색과 안정적인 학습 사이의 균형 | 과제별 차이의 원인을 실험에서 모두 분리한 것은 아님 |

엔트로피 곡선, 훈련 보상, 최종 평가 정확도를 함께 봐야 한다. 학습률 차이도 영향을 줄 수 있어, 보상 상승을 양자화 잡음 하나의 인과 효과로 단정하지 않는다. 근거: [§3–4, Figure 5, Appendix H·J·K](https://arxiv.org/html/2510.11696v1).

## 발표자료 구성

아래 범위는 34장 PPTX와 그 PDF의 실제 순서다. HTML도 같은 순서로 구성했다.

| 페이지 | 내용 |
|---|---|
| 1–4 | 논문 핵심, QeRL 구성 요소, 사용 모델과 비교 방식 |
| 5–7 | 강화학습·GRPO 기초, LoRA의 수식과 학습 파라미터 |
| 8–13 | NVFP4와 커널, 전체 구조, 엔트로피와 탐색의 인과 설명 |
| 14–17 | AQN의 RMSNorm 잡음, 감쇠식, 가중치 관점의 해석, LoRA와 비교 |
| 18–24 | 학습 전후와 방식별 비교, GSM8K·BigMath, AQN의 모델별 효과 |
| 25–30 | 잡음·rank 실험, 저장 크기, 학습·롤아웃 속도, 단일 GPU 학습 |
| 31–34 | 학습률의 영향, 관측과 원인 해석의 구분, 결론과 원문 |

## 한계와 비판적 질문

- Qwen2.5-Instruct와 수학 추론 중심의 결과다. 다른 모델 계열·과제로의 일반화는 별도로 검증해야 한다.
- AQN을 추가해도 모든 과제와 모델 크기에서 성능이 오르지는 않는다.
- 같은 학습률·생성 예산과 여러 random seed로 비교하면, 탐색 효과와 설정 차이의 기여를 더 명확히 분리할 수 있다.
- 저장 크기 감소율, 롤아웃 처리량 배수, 전체 학습 시간, GPU 메모리를 각각 구분해 측정해야 한다.
- 잡음 스케줄의 표기 차이는 발표자 노트에 기록했다. 재현할 때는 사용한 코드 버전과 실제 설정도 함께 고정해야 한다.

## 관련 연구 기록

- [NVFP4 Pretraining & QAD](../2026-08-14-nvfp4-qad/README.md): 저정밀 학습과 양자화 후 정확도 복구를 다룬다.
- [Pretraining Large Language Models with NVFP4](../2026-08-27-pretraining-llms-nvfp4/README.md): Native FP4 사전학습의 안정화 방법을 다룬다.
- [Quantized Reasoning Models](../2026-09-06-quantized-reasoning-overthinking/README.md): PTQ가 추론 중 재고찰과 종료 판단에 주는 영향을 다룬다. 이번 리뷰는 양자화 상태에서의 강화학습과 탐색 조절에 초점을 둔다.

[← 전체 연구 목록으로 돌아가기](../../README.md)
