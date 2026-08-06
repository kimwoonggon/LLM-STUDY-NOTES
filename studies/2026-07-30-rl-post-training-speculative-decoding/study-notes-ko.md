# 논문 발표 사전학습 노트

## Accelerating RL Post-Training Rollouts via System-Integrated Speculative Decoding

- 저자: Hayate Iso et al.
- 공개일: 2026-04-29
- 논문: https://arxiv.org/abs/2604.26779
- 한 문장 요약:

> NeMo-RL의 rollout engine인 vLLM에 speculative decoding을 통합하고, MegatronLM learner와 policy/draft weight를 동기화하여, 현재 policy의 출력 분포와 학습 의미는 유지하면서 reasoning RL의 rollout 생성을 실제로 가속한 시스템 논문이다.

---

## 1. 발표 전에 반드시 기억할 핵심 5가지

1. 이 논문의 새로움은 speculative decoding 알고리즘 자체가 아니다.
   - 핵심은 이를 계속 변하는 RL policy 안에 실제로 통합했다는 점이다.
   - NeMo-RL, vLLM, MegatronLM, GRPO, weight synchronization이 하나의 시스템으로 연결된다.

2. Reasoning RL의 큰 병목은 backward가 아니라 rollout generation일 수 있다.
   - 실제 autoregressive baseline에서 generation이 전체 step 시간의 65-72%였다.

3. Draft는 최종 답을 결정하지 않는다.
   - Draft는 여러 토큰을 빠르게 제안한다.
   - 현재 policy인 verifier가 rejection sampling으로 최종 토큰 분포를 결정한다.
   - GRPO log-probability와 loss도 verifier policy를 기준으로 계산한다.

4. 가장 중요한 실제 결과는 8B synchronous end-to-end 1.35-1.41배다.
   - Generation만 보면 1.5-1.8배다.
   - Async 실제 실험은 1.24배다.
   - 235B의 약 2.5배는 proprietary simulator 기반 projection이다.

5. Acceptance length가 높다고 반드시 빠른 것은 아니다.
   - 더 긴 draft는 더 많은 토큰을 맞혀도 overhead 때문에 느려질 수 있다.
   - n-gram drafting은 acceptance가 2 이상인데도 autoregressive보다 느렸다.

---

## 2. 이 논문은 어떤 문제를 해결하는가?

### 2.1 RL post-training의 기본 흐름

Reasoning 모델을 GRPO로 학습한다고 생각해 보자.

1. Prompt를 준비한다.
2. 현재 policy가 prompt마다 여러 response를 생성한다.
3. 정답 verifier나 reward function으로 response를 채점한다.
4. 그룹 내 상대 reward로 advantage를 계산한다.
5. 현재 policy의 log-probability를 이용해 policy를 업데이트한다.

이때 2번의 response 생성이 rollout generation이다.

지도학습은 이미 존재하는 정답 데이터를 읽으면 되지만, RL은 현재 모델이 학습 데이터를 계속 새로 만들어야 한다. Reasoning response가 길어지고, 한 prompt당 여러 response를 생성하며, agent가 여러 tool call과 환경 step을 수행하면 rollout 비용이 매우 커진다.

### 2.2 왜 기존 autoregressive generation이 느린가?

일반적인 LLM은 한 번의 forward pass에서 다음 토큰 하나를 생성한다.

```text
현재 문맥 -> 큰 policy forward -> token 1
token 1 포함 문맥 -> 큰 policy forward -> token 2
token 2 포함 문맥 -> 큰 policy forward -> token 3
...
```

긴 response를 만들려면 큰 모델의 forward pass를 순차적으로 반복해야 한다. GPU는 병렬 계산에 강하지만, 다음 토큰이 나오기 전에는 그다음 위치를 확정할 수 없다는 순차성이 병목이다.

---

## 3. 왜 중요한 논문인가?

### 3.1 학습 병목의 위치가 바뀌었다

논문의 실제 8B baseline에서 generation 비중은 다음과 같다.

| Workload | 전체 step | Generation | 비중 |
|---|---:|---:|---:|
| RL-Think | 185.3초 | 133.6초 | 약 72.1% |
| RL-Zero | 151.2초 | 100.0초 | 약 66.1% |

즉 optimizer나 backward만 빨라져도 전체 학습 시간 대부분은 그대로 남을 수 있다.

### 3.2 Throughput과 learning effectiveness를 분리해서 생각한다

논문은 learning speed를 다음처럼 본다.

```text
learning speed = effectiveness x throughput
```

- Throughput: 단위 시간당 수행하는 rollout과 update의 양
- Effectiveness: 각 rollout에서 얻는 유용한 학습 신호

비동기 실행, replay, off-policy learning, low-precision rollout도 throughput을 높일 수 있다. 그러나 policy lag, sampling distribution mismatch, importance sampling correction 같은 문제가 생길 수 있다.

Speculative decoding의 목표는 verifier policy의 sampling distribution을 유지하면서 throughput만 높이는 것이다. 이것이 이 논문의 가장 중요한 철학이다.

### 3.3 학습과 inference system을 함께 이해하게 해 준다

이 논문을 이해하려면 다음을 동시에 봐야 한다.

- RL algorithm: GRPO, advantage, on-policy semantics
- Training backend: MegatronLM
- Rollout backend: vLLM
- Inference optimization: speculative decoding
- Distributed system: weight synchronization, colocated/non-colocated execution
- Pipeline: synchronous/asynchronous RL

따라서 단순한 알고리즘 논문보다 실제 LLM 학습 시스템의 전체 구조를 공부하기 좋다.

---

## 4. Speculative decoding의 직관

### 4.1 초안 작성자와 최종 편집자

- Draft model: 작고 빠른 초안 작성자
- Verifier 또는 target model: 현재 policy이며 최종 결정을 내리는 편집자

예를 들어 draft length가 3이면 draft가 다음 세 토큰을 먼저 제안한다.

```text
Draft proposal:  "정답은" "42" "이다"
Verifier check:   accept  accept  reject
Correction:                      "입니다"
```

Verifier는 짧은 continuation의 여러 위치를 한 번에 평가할 수 있다. 앞에서부터 연속으로 맞는 토큰들을 받아들이면, 큰 policy forward 한 번으로 여러 토큰을 생성한 효과를 얻는다.

### 4.2 Target은 Draft 제안을 어떻게 검증하는가?

현재 확정된 문장이 `철수는`이고 EAGLE-3가 `영희를 / 좋아한다 / .`을 제안했다고 하자.

개념적으로 Target은 다음 문장을 검사한다.

```text
철수는 | 영희를 좋아한다 .
         └ Draft proposal
```

실제 KV-cache 구현에서는 `철수는`을 다시 계산하지 않는다.

```text
past_key_values = Target_KV("철수는")
input_ids       = ["영희를", "좋아한다", "."]
```

Target의 causal forward 한 번은 위치별 다음-token logits를 만든다.

| Target이 조건으로 사용하는 문맥 | Target 예측 | Draft 제안 | 판단 |
|---|---|---|---|
| `철수는` | `영희를` | `영희를` | 승인 |
| `철수는 영희를` | `좋아한다` | `좋아한다` | 승인 |
| `철수는 영희를 좋아한다` | `.` | `.` | 승인 |

Causal mask 때문에 각 위치는 뒤쪽 Draft token을 볼 수 없다. 따라서 미래 정답을 훔쳐보는 것이 아니라, teacher-forcing 형태로 여러 위치의 logits를 GPU에서 묶어 계산하는 것이다.

Target과 Draft는 보통 서로 다른 모델이며 KV cache도 공유하지 않는다.

```text
Target KV = Qwen3 weight로 계산한 KV
Draft KV  = EAGLE-3 weight로 계산한 별도 상태
```

Draft의 의미는 Target을 없애는 데 있지 않다. 비싼 Target을 token마다 순차 호출하는 횟수를 줄이는 데 있다.

```text
Autoregressive:
Target 1-token forward × 3회

Speculative:
작은 Draft의 proposal
+ Target 3-token block verification × 1회
```

LLM의 1-token decode는 거대한 weight와 KV를 반복해서 읽는 memory-bound 작업인 경우가 많다. 여러 위치를 한 block으로 처리하면 weight 읽기와 kernel-launch 비용을 분산하고 GPU 병렬성을 더 활용할 수 있다. 다만 Draft 생성과 긴 block 검증에도 비용이 있으므로 항상 빨라지는 것은 아니다.

### 4.3 분포를 어떻게 보존하는가?

한 토큰에 대한 단순화된 설명은 다음과 같다.

- Verifier distribution: `p_theta(x)`
- Draft distribution: `q_phi(x)`
- Draft가 제안한 토큰을 받아들일 확률:

```text
a(x) = min(1, p_theta(x) / q_phi(x))
```

- 거절되면 verifier와 draft의 남은 확률 차이에서 다시 표본을 뽑는다.

```text
r(x) proportional to max(p_theta(x) - q_phi(x), 0)
```

이 acceptance와 correction을 합치면 최종 표본은 verifier distribution `p_theta`에서 직접 생성한 것과 같은 분포를 따른다.

따라서 draft가 부정확하면 주로 속도가 나빠진다. 올바른 rejection sampling과 verifier 구현을 전제로 하면 최종 sampling distribution을 draft가 바꾸지 않는다.

---

## 5. RL에 통합할 때 일반 inference와 무엇이 다른가?

일반 inference serving에서는 target model weight가 고정되어 있다. RL에서는 learner가 매 step policy를 바꾼다.

### 먼저 용어를 정확히 구분하자

이 논문에서는 같은 Qwen3 Policy가 세 가지 이름으로 불린다.

```text
GRPO Policy = 학습 대상 모델
            = rollout의 Target model
            = Draft proposal을 검사하는 Verifier
```

여기서 Verifier는 수학 답을 채점하는 reward verifier가 아니다.

```text
Policy/Target/Verifier: 다음 token 분포를 결정
Reward function:        완성된 답의 품질과 정답 여부를 채점
Reference model:        KL 기준을 제공
Draft model:            미래 token 후보를 빠르게 제안
```

GRPO라서만 가능한 구조는 아니다. 현재 Policy 분포에서 rollout을 생성해야 하는 PPO 등 다른 policy-gradient RL에서도 Policy가 speculative Target이 될 수 있다.

이 때문에 다음 네 가지가 필요하다.

### 5.1 Policy weight synchronization

MegatronLM learner에서 업데이트된 policy weight를 vLLM rollout engine에 전달해야 한다.

Rollout engine이 오래된 weight를 사용하면 현재 policy가 아닌 stale policy에서 trajectory가 생성되고 policy lag가 커질 수 있다.

예를 들어 GRPO update 뒤에는 다음과 같은 불일치가 생길 수 있다.

```text
MegatronLM learner: Policy version 11
vLLM rollout:       Policy version 10
```

이 상태에서 생성하면 version 10의 rollout을 version 11 기준으로 학습하게 되어 on-policy 의미와 log-prob 계산이 어긋난다. 따라서 learner의 최신 Policy weight를 vLLM의 Target/Verifier에 refit 또는 전송해야 한다.

Online Draft adaptation을 사용한다면 최신 Draft weight도 rollout engine으로 전달한다. `Policy/Draft weight sync`는 두 모델의 weight를 서로 같게 만든다는 뜻이 아니라, Policy와 Draft 각각의 최신 checkpoint를 그것을 실행하는 worker에 전달한다는 뜻이다. Async에서는 이 동기화를 일부러 늦추되 `policy lag`라는 허용 범위 안에서 관리한다.

### 5.2 Draft coherence

Policy가 바뀌는데 draft가 고정되어 있으면 두 distribution의 차이가 커질 수 있다. 그러면 acceptance length가 줄고 speculative decoding의 속도 이득이 사라진다.

이를 완화하는 두 방법이 있다.

- Offline draft: 잘 초기화한 draft를 고정해서 사용
- Online draft adaptation: RL 중에 현재 policy를 teacher로 draft를 계속 업데이트

### 5.3 Verifier-side log-probability recomputation

GRPO의 probability ratio, KL penalty, policy loss는 draft가 아니라 verifier policy의 확률을 사용해야 한다.

Draft는 trajectory를 빨리 생성하기 위한 proposal mechanism일 뿐이다. Draft log-probability를 policy loss에 사용하면 최적화 대상 자체가 바뀐다.

### 5.4 Draft gradient와 policy gradient 분리

Online EAGLE-3를 학습할 때 MegatronLM forward pass에서 얻은 hidden states와 verifier output을 draft supervision에 재사용한다.

하지만 draft loss의 gradient가 GRPO policy gradient를 변경하면 안 된다. 그래서 cached feature와 teacher output을 gradient-detached pathway로 draft에 전달한다.

```text
L_total = L_GRPO(theta) + lambda * L_draft(phi; stopgrad(policy outputs))
```

이 수식은 두 loss가 함께 계산될 수 있지만, draft loss가 policy parameter theta를 업데이트하지 않도록 분리한다는 뜻이다.

---

## 6. 전체 시스템 구조

```text
Prompt / Environment
        |
        v
vLLM rollout engine
  - Draft: 여러 토큰 제안
  - Current policy verifier: 제안 검증
        |
        v
Verifier-exact rollout trajectory
        |
        v
MegatronLM learner
  - Current-policy log-prob recomputation
  - Reward / advantage / GRPO loss
  - Optional cached hidden states for draft loss
        |
        v
Policy update + optional draft update
        |
        +-------- weight refit/sync --------> vLLM
```

논문의 핵심 공헌은 이 전체 loop를 NeMo-RL 안에서 synchronous와 asynchronous 방식 모두 지원하도록 구현한 것이다.

---

## 7. 속도 상한을 이해하는 수식

Synchronous RL step은 다음처럼 분해된다.

```text
T_step = T_data + T_prepare + T_gen + T_logprob + T_train
```

- `T_data`: 데이터 준비
- `T_prepare`: weight synchronization, rollout backend 준비
- `T_gen`: rollout generation
- `T_logprob`: 현재 policy의 log-probability 재계산
- `T_train`: advantage 계산과 policy optimization

Speculative decoding이 직접 줄이는 것은 `T_gen` 중에서도 주로 autoregressive decode 부분이다. Prefill, log-prob recomputation, training은 그대로 남는다.

논문이 제시한 이상적 상한은 다음과 같다.

```text
S_step <= 1 / (R_gen / alpha + (1 - R_gen))
```

- `R_gen = T_gen / T_step`
- `alpha`: speculation step 하나가 평균적으로 생성하는 token 수인 mean acceptance length

### 실제 수치로 보는 의미

RL-Zero:

- `R_gen = 100.0 / 151.2 = 약 0.661`
- `alpha = 3.32`
- 이상적 상한은 약 1.86배
- 실제 end-to-end는 1.41배

RL-Think:

- `R_gen = 133.6 / 185.3 = 약 0.721`
- `alpha = 2.77`
- 이상적 상한은 약 1.85배
- 실제 end-to-end는 1.35배

차이가 생기는 이유는 다음과 같다.

- Draft model 실행 overhead
- Verification overhead
- Prefill 시간
- Batching과 GPU utilization
- Weight synchronization과 backend 준비
- 가속되지 않는 log-prob와 training 단계

따라서 acceptance length를 generation speedup이나 end-to-end speedup과 동일시하면 안 된다.

---

## 8. 사용한 모델, 데이터, 알고리즘, 하드웨어

### 8.1 Policy model

| 설정 | 모델 | 의미 |
|---|---|---|
| RL-Think | Qwen3-8B | 이미 reasoning이 가능한 모델에서 thinking trace를 더 개선 |
| RL-Zero | Qwen3-8B-Base | Base model에서 직접 RL을 시작 |

RL-Think는 이미 사고 능력이 있는 모델을 계속 학습하는 설정이고, RL-Zero는 base model이 RL을 통해 reasoning behavior를 만들어 가는 설정이다.

`RL-Think`와 `RL-Zero`는 알고리즘 이름이 아니라 논문이 붙인 두 실험 설정 이름이다.

```text
RL-Think:
이미 instruction/reasoning post-training을 받은 Qwen3-8B
→ 기존 thinking trace를 GRPO로 개선

RL-Zero:
언어 pretraining만 된 Qwen3-8B-Base
→ 별도의 reasoning post-training 없이 GRPO를 시작
```

`Zero`는 무작위 초기화나 지식이 0이라는 뜻이 아니다. Pretrained base model에서 reasoning RL을 바로 시작한다는 뜻이다.

RL-Zero의 acceptance가 RL-Think보다 높은 정확한 원인은 논문이 별도 ablation으로 증명하지 않았다. 다만 EAGLE-3와 n-gram 모두 RL-Zero에서 acceptance가 높다는 점은, RL-Zero rollout이 평균적으로 더 짧거나 반복적이고 다음 token을 예측하기 쉬웠다는 해석을 뒷받침한다. Acceptance가 높다는 것은 Draft가 Policy를 맞히기 쉽다는 뜻이지, reasoning 품질이 더 높다는 뜻은 아니다.

### 8.2 Training과 validation 데이터

| 용도 | 데이터 |
|---|---|
| RL training | DAPO-Math-17K |
| Validation | AIME-2024 accuracy |

논문의 actual experiment는 수학 reasoning domain에 집중한다.

### 8.3 Draft

- Main draft mechanism: EAGLE-3
- 기본 draft length: `k = 3`
- Main experiment: offline initialization 후 RL 중 draft weight 고정
- 비교 baseline: autoregressive decoding, n-gram drafting
- 추가 분석: draft initialization, draft length, online adaptation

#### Qwen3와 EAGLE-3의 관계

```text
Qwen3-8B        = 실제 학습 대상 Policy이자 최종 Target/Verifier
Qwen3용 EAGLE-3 = Qwen3의 미래 token을 미리 제안하는 전용 보조 Draft
```

EAGLE-3는 Qwen3를 대체하지 않는다. Qwen3의 여러 layer hidden feature를 융합해 미래 token을 직접 예측하고, Qwen3가 그 proposal을 다시 검증한다. 특정 Target의 hidden dimension, tokenizer, vocabulary, feature distribution에 맞춰 학습되므로 Qwen3-8B용 Draft를 다른 크기의 Qwen이나 Llama에 그대로 붙일 수 없다.

논문은 사용한 EAGLE-3 checkpoint의 정확한 parameter 수를 공개하지 않았다. 크기를 단정해서 발표하면 안 된다. 규모 감각을 위한 공개 Qwen3-8B EAGLE-3 예시는 1-layer, 약 350M parameters로 소개되지만, 이는 논문 checkpoint의 공식 수치가 아니다: https://huggingface.co/thoughtworks/Qwen3-8B-Eagle3

#### Native MTP head와의 차이

Native MTP는 Target 내부에 사전학습 때부터 포함된 multi-token prediction head다.

```text
hidden("철수는")
  ├─ 다음 token 예측   → 영희를
  ├─ 2번째 미래 예측   → 좋아한다
  └─ 3번째 미래 예측   → .
```

```text
EAGLE-3: Target 외부의 별도 Draft module과 checkpoint
Native MTP: Target 내부에 내장된 auxiliary prediction head
```

Native MTP가 없는 pretrained 모델에는 EAGLE-3 같은 external path가 필요하다. 이 논문은 두 경로를 시스템적으로 지원하지만 실제 8B 핵심 실험은 EAGLE-3 경로에 집중한다.

EAGLE-3를 선택한 이유는 native MTP head가 없는 pretrained model에도 붙일 수 있으며, moving policy와 정합성을 유지해야 하는 더 어려운 일반 경로이기 때문이다.

### 8.4 RL algorithm과 system

- RL algorithm: GRPO
- Rollout backend: vLLM
- Training backend: MegatronLM
- Orchestration/framework: NeMo-RL

### 8.5 하드웨어

- 8개 GB200 NVL72 node
- Node당 4개 GB200 GPU 사용
- 총 32개 GB200 GPU
- GPU당 186 GB HBM3E
- Fifth-generation NVLink

---

## 9. Main result

### 9.1 Step time breakdown

| Stage | RL-Think AR | RL-Think Spec | RL-Zero AR | RL-Zero Spec |
|---|---:|---:|---:|---:|
| Data | 0.3s | 0.2s | 0.2s | 0.2s |
| Prepare | 2.1s | 1.6s | 1.9s | 2.1s |
| Generation | 133.6s | 87.0s | 100.0s | 56.6s |
| Log-prob | 17.9s | 18.1s | 17.8s | 18.1s |
| Training | 31.4s | 30.5s | 31.3s | 30.5s |
| Overall | 185.3s | 137.4s | 151.2s | 107.5s |

### 9.2 정확한 해석

RL-Think:

- Generation speedup: `133.6 / 87.0 = 약 1.54배`
- End-to-end speedup: `185.3 / 137.4 = 약 1.35배`
- Step time 감소율: 약 25.9%

RL-Zero:

- Generation speedup: `100.0 / 56.6 = 약 1.77배`
- End-to-end speedup: `151.2 / 107.5 = 약 1.41배`
- Step time 감소율: 약 28.9%

주의:

> 1.41배 speedup은 시간이 41% 감소했다는 뜻이 아니다. 새 시간은 기존의 `1 / 1.41 = 약 71%`이므로 약 29% 감소다.

### 9.3 학습 품질

AIME-2024 validation accuracy는 autoregressive와 EAGLE-3에서 거의 동일한 trajectory를 보였다.

- RL-Think: 약 0.60에서 약 0.70
- RL-Zero: 약 0.03에서 약 0.33

논문이 보여 준 것은 해당 8B 수학 workload에서 validation trajectory가 구분되지 않았다는 것이다. 모든 task와 모든 system에서 품질이 동일하다는 광범위한 실험적 증명은 아니다.

---

## 10. Ablation study와 운영상 교훈

이 논문의 ablation은 단순한 부가 실험이 아니라, 실제 배포 시 무엇을 조절해야 하는지 알려 주는 핵심 부분이다.

### 10.1 EAGLE-3와 n-gram drafting 비교

| Workload | Method | Acceptance length | Generation latency | Speedup |
|---|---|---:|---:|---:|
| RL-Zero | Autoregressive | - | 100.0s | 1.0배 |
| RL-Zero | n-gram | 2.47 | 140.2s | 0.7배 |
| RL-Zero | EAGLE-3 | 3.32 | 56.6s | 1.8배 |
| RL-Think | Autoregressive | - | 133.6s | 1.0배 |
| RL-Think | n-gram | 2.05 | 262.9s | 0.5배 |
| RL-Think | EAGLE-3 | 2.77 | 87.0s | 1.5배 |

#### n-gram Draft는 무엇인가?

별도 신경망 없이 현재 문맥에서 반복된 token pattern을 찾아 그 뒤 token을 복사해 제안한다.

```text
과거 문맥: 철수는 영희를 좋아한다 .
현재 문맥: 민수도 영희를 ...

n-gram proposal: 좋아한다 / .
```

코드나 반복 문구에는 잘 맞을 수 있지만 새로운 수학 reasoning 전개에는 취약하다.

#### Acceptance length는 무엇인가?

Acceptance rate와 다른 값이다.

```text
Acceptance rate   = 제안 token 중 승인된 비율
Acceptance length = Target 검증 한 번당 평균 몇 token을 전진했는가
```

`k = 3`이면 첫 proposal이 틀려도 Target correction 한 token은 생성하므로 보통 최소 1 token 전진한다. 세 proposal이 모두 맞으면 bonus token까지 최대 `k + 1 = 4` token 전진할 수 있다. 따라서 EAGLE-3의 `3.32`는 “검증 한 번당 평균 3.32 token을 출력했다”는 뜻이며 332%라는 뜻이 아니다.

해석:

- n-gram도 평균 2개 이상의 token을 생성한다.
- 그런데 verification과 proposal overhead 때문에 baseline보다 느리다.
- Positive acceptance만으로는 충분하지 않다.
- 최종 판단 지표는 generation latency와 end-to-end step time이어야 한다.

### 10.2 Draft initialization

`k = 3`, offline draft 조건이다.

| Initialization | RL-Zero acceptance | RL-Zero speedup | RL-Think acceptance | RL-Think speedup |
|---|---:|---:|---:|---:|
| UltraChat/Magpie | 2.88 | 1.51배 | 2.40 | 1.19배 |
| DAPO in-domain | 3.32 | 1.77배 | 2.77 | 1.53배 |

해석:

- Generic chat draft보다 실제 rollout domain에 맞는 draft가 훨씬 좋다.
- Draft의 일반적인 언어 능력보다 target rollout distribution과의 정합성이 중요하다.
- 수학 RL에는 수학 reasoning response로 draft를 초기화하는 편이 좋다.
- Code agent라면 code와 tool-call trajectory를 닮은 데이터가 필요할 가능성이 높다.

### 10.3 Draft length

| Draft length k | RL-Zero acceptance | RL-Zero speedup | RL-Think acceptance | RL-Think speedup |
|---:|---:|---:|---:|---:|
| 3 | 3.32 | 1.77배 | 2.77 | 1.53배 |
| 5 | 4.35 | 1.44배 | 3.23 | 0.84배 |
| 7 | 5.06 | 1.21배 | 3.48 | 0.71배 |

`k`를 크게 했을 때의 핵심은 절대 acceptance가 아니라 proposal 대비 유효 진전량이다.

```text
RL-Think
k=3: 3개 제안 → 평균 2.77 token 전진
k=5: 5개 제안 → 평균 3.23 token 전진
k=7: 7개 제안 → 평균 3.48 token 전진
```

`k=3`에서 `k=7`로 proposal 계산은 133% 늘었지만 전진량은 약 26%만 늘었다. EAGLE-3는 더 많은 token을 만들고 Target은 더 긴 block의 logits와 임시 KV를 계산한다. 중간 rejection 뒤의 proposal과 KV는 폐기된다. 추가 계산이 추가 accepted token보다 커지면 speedup이 1 아래로 내려간다.

Autoregressive token당 비용을 1로 정규화하면 RL-Think의 상대 비용은 다음과 같다.

```text
AR:    1.00
k=3:   1 / 1.53 = 0.65
k=5:   1 / 0.84 = 1.19
k=7:   1 / 0.71 = 1.41
```

따라서 `k=7`은 더 많이 받아들이면서도 token당 실제 비용은 AR보다 약 41% 컸다.

가장 중요한 결과:

- `k`가 커질수록 acceptance는 증가했다.
- 하지만 speedup은 계속 감소했다.
- RL-Think에서는 `k >= 5`가 autoregressive보다 느렸다.

이유:

- Draft token을 더 많이 생성해야 한다.
- Verifier가 더 긴 proposal을 검사해야 한다.
- 더 많은 accepted token이 추가 계산량을 보상하지 못할 수 있다.

따라서 optimal `k`는 acceptance를 최대화하는 값이 아니라 end-to-end latency를 최소화하는 값이다.

### 10.4 Online draft adaptation

| Variant | RL-Zero acceptance | RL-Zero speedup | RL-Think acceptance | RL-Think speedup |
|---|---:|---:|---:|---:|
| UltraChat offline | 2.88 | 1.51배 | 2.40 | 1.19배 |
| UltraChat online | 3.04 | 1.63배 | 2.55 | 1.26배 |
| DAPO offline | 3.32 | 1.77배 | 2.77 | 1.53배 |
| DAPO online | 3.29 | 1.78배 | 2.74 | 1.52배 |

해석:

- 약한 UltraChat initialization에서는 online adaptation이 의미 있게 도움을 준다.
- 이미 잘 맞는 DAPO initialization에서는 추가 이득이 거의 없다.
- Online training은 항상 성능을 높이는 만능 방법이 아니다.
- Distribution mismatch가 커질 때 draft를 보호하는 보험에 가깝다.

### 10.5 Synchronous와 asynchronous 실행

Async 실험 조건:

- Workload: RL-Think
- Policy lag: 1
- 16-node non-colocated
- Generation: 12 nodes
- Training: 4 nodes

결과:

- Exposed generation time: 10.4초에서 0.6초
- Effective step time: 75.0초에서 60.5초
- End-to-end speedup: 1.24배

`Exposed generation time`은 generation 전체 시간이 아니다. Async overlap으로 숨기지 못해 learner가 rollout을 기다리는 시간이다.

```text
Generation: ─────────────────────────
Training:       ─────────────────
                                └ 기다리는 꼬리 = exposed generation
```

Baseline에서는 이 대기 꼬리가 10.4초였고 speculative decoding 후 0.6초가 되었다. 대기 시간은 거의 사라졌지만 약 60초의 training/log-prob/통신 critical path는 남기 때문에 effective step은 75.0초에서 60.5초, 즉 `75.0 / 60.5 = 1.24배`만 개선된다.

왜 synchronous보다 배율이 작은가?

Async pipeline은 generation을 log-prob와 training 뒤에 이미 숨긴다. Speculative decoding이 줄인 시간 중 critical path에 노출되어 있던 부분만 전체 step speedup으로 나타난다.

그래도 두 기술은 보완 관계다.

- Async: 남은 일을 다른 stage와 겹쳐서 숨긴다.
- Speculative decoding: rollout 자체의 계산량과 latency를 줄인다.

---

## 11. 235B 결과를 어떻게 해석해야 하는가?

### 11.1 Actual experiment가 아니다

235B 결과는 proprietary GPU performance simulator로 계산한 projection이다.

`Proprietary simulator`는 회사가 내부적으로 보유한 비공개 성능 모의실험 프로그램이라는 뜻이다.

```text
Proprietary = 소유권이 있는 비공개
Simulator   = 실제 실행 대신 성능을 계산하는 모의실험기
```

GPU compute, HBM bandwidth, NVLink, kernel, sharding, batch와 response-length 분포를 모델링해 예상 시간을 계산한다. 실제 235B RL run이 아니며 내부 코드와 calibration이 공개되지 않아 다른 연구자가 완전히 재현하기 어렵다.

Simulation 조건에는 다음이 포함된다.

- Qwen3 family
- Rollout batch size 4096
- FP8
- 최대 2048 GB200 GPUs
- Model sharding, GPU compute, memory hierarchy, interconnect 모델
- Long-tailed response length traffic model

저자는 simulation 결과를 absolute prediction이 아니라 opportunity envelope와 경향으로 해석하라고 명시한다.

### 11.2 주요 projection

Qwen3-235B-A22B, 512 GPUs:

- Favorable draft/acceptance 조합에서 rollout 최대 6.49배
- Non-generation stage 때문에 end-to-end 최대 2.22배

Qwen3-235B-A22B, 2048 GPUs, policy lag 2:

- Rollout 약 3.5배
- End-to-end 약 2.5배

#### 숫자로 이해하기

512-GPU synchronous heatmap의 최대 조합을 전체 100초로 단순화하면 다음과 같다.

```text
Baseline: generation 65초 + 기타 단계 35초 = 100초
Spec:     65 / 6.49 + 35 ≈ 45초
End-to-end speedup: 100 / 45 ≈ 2.22배
```

2048 GPUs, lag 2는 동일 조건이 아니다. Figure 4는 `k = 5`, acceptance 4를 고정하고 scale과 lag를 바꾼 별도 simulation이다. Frontier-scale 조건에서 generation share가 약 84%라고 단순화하면 다음처럼 이해할 수 있다.

```text
Baseline: generation 84초 + 기타 단계 16초 = 100초
Spec:     84 / 3.5 + 16 = 40초
End-to-end speedup: 100 / 40 = 2.5배
```

따라서 `6.49 → 2.22`와 `3.5 → 2.5`를 같은 실험의 숫자로 비교하면 안 된다. Draft/acceptance, GPU scale, policy lag, baseline stage 비중이 서로 다르다.

### 11.3 발표에서의 올바른 표현

잘못된 표현:

> 이 논문은 235B 모델 학습을 실제로 2.5배 가속했다.

올바른 표현:

> 8B 실제 실험에서는 synchronous end-to-end 최대 1.41배를 측정했고, 235B의 유리한 대규모 배치 조건에서는 proprietary simulator가 약 2.5배의 end-to-end 기회를 전망했다.

---

## 12. 무엇을 개선했는가?

### 알고리즘 측면

- 새로운 RL objective를 제안한 것은 아니다.
- Verifier-exact speculative sampling을 RL rollout에 사용한다.

### 시스템 측면

- vLLM speculative decoding을 NeMo-RL rollout path에 통합
- MegatronLM learner와 rollout engine 사이 policy weight synchronization
- EAGLE-3 draft weight의 offline/online 관리
- Cached hidden states와 verifier outputs 재사용
- Draft gradient를 policy gradient에서 detach
- Verifier-side log-prob, KL, policy loss 유지
- Synchronous와 asynchronous RL 모두 지원
- Stage-level latency와 acceptance telemetry로 실제 speedup 측정
- External EAGLE-3 path와 native MTP path를 모두 고려

### 결과 측면

- 8B generation latency 1.5-1.8배 개선
- 8B synchronous end-to-end 1.35-1.41배 개선
- Async effective step 1.24배 개선
- AIME validation trajectory 유지

---

## 13. 이 논문의 강점

### 13.1 Production-grade integration을 다룬다

논문이 단순히 `speculative_config` 옵션을 켜는 수준에 머물지 않는다. Moving policy, draft coherence, log-prob recomputation, weight refit, asynchronous critical path까지 다룬다.

### 13.2 실제 stage time을 보여 준다

Generation만 빠르다고 주장하지 않고 Data, Prepare, Generation, Log-prob, Training 시간을 모두 공개한다. 이를 통해 end-to-end ceiling을 이해할 수 있다.

### 13.3 Ablation이 실무적이다

- 어떤 데이터로 draft를 초기화해야 하는가?
- Draft length를 늘리면 좋은가?
- Online adaptation이 항상 필요한가?
- Async와 같이 쓰면 어떤가?

실제 운영자가 선택해야 할 질문에 직접 답한다.

### 13.4 Evidence와 projection을 구분한다

저자들은 simulator 결과를 opportunity envelope라고 명시한다. 리뷰어도 이 구분을 유지해야 한다.

---

## 14. 한계와 비판적으로 볼 지점

### 14.1 실제 대규모 모델 실험이 없다

- Actual result는 8B, 32 GPUs다.
- 235B, 최대 2048 GPUs 결과는 simulation이다.

### 14.2 Actual task가 수학 reasoning에 한정된다

- Training: DAPO-Math-17K
- Validation: AIME-2024

Code RL, multi-turn tool agent, web agent, multimodal RL에서 같은 효과가 나는지는 실제로 검증하지 않았다.

### 14.3 Simulator가 proprietary다

대규모 projection의 내부 구현과 calibration을 완전히 재현하기 어렵다. 절대 수치보다 추세를 보는 것이 안전하다.

### 14.4 Baseline 비교 범위가 제한적이다

실제 main experiment는 AR, n-gram, EAGLE-3 비교가 중심이다. FastGRPO, ReSpec, FP8 rollout, 다른 RL framework와 동일 조건의 직접 end-to-end 비교는 없다.

### 14.5 품질 검증 범위가 제한적이다

AIME validation trajectory가 겹친 것은 중요한 evidence다. 그러나 더 다양한 benchmark, reward hacking, output diversity, long-horizon agent behavior까지 평가하지는 않았다.

### 14.6 Draft 준비 비용이 충분히 분석되지 않는다

In-domain EAGLE-3 draft를 offline으로 학습하는 비용, checkpoint 관리 비용, online training의 memory/communication overhead가 전체 프로젝트 비용에서 언제 amortize되는지 더 분석할 수 있다.

### 14.7 최적 설정은 workload 의존적이다

`k = 3`이 실험에서 가장 좋았지만 모든 모델과 task에 보편적인 값은 아니다. Batch size, response length, GPU, model architecture, acceptance distribution에 따라 다시 profiling해야 한다.

---

## 15. 관련 논문 지도

### 15.1 Speculative decoding의 기초

#### Leviathan et al., Fast Inference from Transformers via Speculative Decoding

- https://arxiv.org/abs/2211.17192
- 작은 approximation model이 여러 token을 제안하고 target model이 정확한 sampling distribution을 유지하며 검증하는 기본 아이디어를 제시한다.

#### Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling

- https://arxiv.org/abs/2302.01318
- Modified rejection sampling으로 target distribution을 보존하는 speculative sampling을 제시한다.

현재 리뷰 논문은 이 lossless inference 아이디어를 moving-policy RL system 안으로 가져온다.

### 15.2 Draft architecture와 verification 개선

#### Medusa

- https://arxiv.org/abs/2401.10774
- 여러 decoding head로 미래 token 후보를 만들고 tree 형태로 검증한다.

#### EAGLE

- https://arxiv.org/abs/2401.15077
- Token만 예측하는 대신 target model의 feature를 활용해 더 정확한 drafting을 시도한다.

#### EAGLE-3

- https://arxiv.org/abs/2503.01840
- Multi-layer feature fusion과 direct token prediction을 사용한다.
- 현재 리뷰 논문의 general external draft path가 EAGLE-3 기반이다.

#### Better & Faster Large Language Models via Multi-token Prediction

- https://arxiv.org/abs/2404.19737
- Pretraining 때 여러 미래 token을 예측하는 auxiliary MTP heads를 학습한다.
- 현재 리뷰 논문의 native MTP path에서 이 head들이 draft 역할을 할 수 있다.

### 15.3 Speculative decoding을 RL에 직접 적용한 연구

#### FastGRPO

- https://arxiv.org/abs/2509.21792
- High-concurrency GRPO에서 speculative decoding 효율이 떨어지는 문제를 다룬다.
- Real-time concurrency에 따라 draft/verification 전략을 조절한다.
- Online draft learning을 사용한다.
- 논문은 2.35-2.72배 end-to-end speedup을 보고하지만 실험 조건이 다르므로 현재 리뷰 논문의 1.41배와 직접 비교하면 안 된다.

#### ReSpec

- https://arxiv.org/abs/2510.26475
- Large batch에서 speedup 감소, stale drafter, policy degradation을 문제로 본다.
- Dynamic configuration, knowledge distillation, reward-weighted draft update를 사용한다.
- Qwen 3B-14B에서 최대 4.5배를 보고하지만 역시 조건과 metric을 확인해야 한다.

현재 리뷰 논문의 차별점:

- NeMo-RL production stack 안의 end-to-end integration
- Verifier-exact training semantics
- Coordinated policy/draft weight synchronization
- Sync와 async 조합 분석
- Deployment scale projection

### 15.4 Async와 pipeline 중심 RL system

#### PipelineRL

- https://arxiv.org/abs/2509.19128
- Generation과 training을 겹치며 in-flight weight update로 policy freshness를 유지한다.
- Speculative decoding이 rollout 자체를 싸게 만든다면, PipelineRL류 방법은 남은 rollout 시간을 training 뒤에 숨긴다.

#### LlamaRL

- https://arxiv.org/abs/2505.24034
- 대규모 distributed asynchronous RL framework다.

### 15.5 Low-precision rollout

#### FP8-RL

- https://arxiv.org/abs/2601.18150
- FP8 rollout으로 compute와 memory traffic을 줄인다.
- Train-inference mismatch를 correction으로 관리한다.

Speculative decoding과의 차이:

- FP8은 numerical precision을 바꾼다.
- Speculative decoding은 verifier distribution을 보존하는 방향으로 generation sequence를 바꾼다.
- 두 방법은 잠재적으로 조합 가능하지만 안정성과 end-to-end 효과를 따로 측정해야 한다.

---

## 16. 발표에서 나올 가능성이 높은 질문과 답변

### Q1. 이 논문의 진짜 novelty는 무엇인가?

Speculative decoding 자체가 아니라 moving-policy RL loop에 verifier-exact하게 통합한 시스템 설계다. Policy/draft weight sync, MegatronLM log-prob recomputation, detached online draft loss, sync/async composition이 핵심이다.

### Q2. Draft model이 틀리면 학습 품질이 떨어지지 않는가?

올바른 speculative rejection sampling에서는 verifier가 최종 distribution을 결정한다. Draft가 틀리면 acceptance가 줄어 속도가 나빠지는 것이 주된 결과다. 다만 구현 오류나 numerical/backend mismatch가 없다는 전제가 필요하다.

### Q3. 왜 rollout 후 log-probability를 다시 계산하는가?

GRPO loss는 현재 verifier policy의 확률을 사용해야 하기 때문이다. Rollout engine의 생성 경로와 learner의 최적화 경로를 분리하고, trainer가 현재 policy 기준 log-prob과 KL을 계산한다.

### Q4. Online draft adaptation은 꼭 필요한가?

아니다. Weak initialization에서는 도움이 되었지만, DAPO in-domain initialization에서는 offline 1.77배와 online 1.78배로 차이가 거의 없었다. Drift가 예상될 때 쓰는 보험에 가깝다.

### Q5. Draft를 길게 하면 왜 느려지는가?

Accepted token 수는 늘어도 draft generation과 verification work가 더 많이 든다. RL-Think는 `k = 5`부터 baseline보다 느려졌다.

### Q6. Async result가 1.24배로 더 작은데 의미가 있는가?

Async는 generation의 상당 부분을 이미 training 뒤에 숨긴다. 따라서 전체 배율은 작아진다. 그러나 exposed generation idle이 10.4초에서 0.6초로 거의 제거되었다는 의미가 있다.

### Q7. 235B 2.5배를 믿어도 되는가?

실측으로 말하면 안 된다. Proprietary simulator의 favorable operating point projection이다. 방향성과 opportunity를 보여 주지만, actual deployment에서 재검증해야 한다.

### Q8. 왜 n-gram은 acceptance가 2 이상인데 느린가?

Acceptance length는 추가 proposal과 verification 비용을 반영하지 않는다. Hardware utilization과 implementation overhead까지 포함한 실제 latency가 중요하다.

### Q9. 어디에 가장 유용한가?

- Generation share가 큰 RL
- Response가 길고 decode-heavy한 reasoning
- 한 prompt당 여러 response를 생성하는 GRPO
- 반복 tool call이 있는 agentic RL
- In-domain draft를 확보할 수 있는 workload

### Q10. 어디에는 덜 유용한가?

- 짧은 response
- Prefill이 지배적인 workload
- Large batch에서 target utilization이 이미 높은 경우
- Draft가 policy와 잘 맞지 않는 경우
- Async pipeline이 generation을 거의 모두 숨기는 경우

### Q11. GRPO Policy와 Target/Verifier는 같은 모델인가?

그렇다. 이 논문에서는 GRPO로 업데이트되는 Qwen3 Policy가 rollout의 Target이자 EAGLE-3 proposal을 검사하는 Verifier다. 단, 수학 답을 채점하는 reward verifier와는 역할이 다르다.

### Q12. EAGLE-3와 Qwen3는 어떤 관계인가?

Qwen3가 본체 Policy/Teacher/Verifier이고, EAGLE-3는 특정 Qwen3의 hidden feature와 출력 분포를 학습한 전용 보조 Draft다. 서로 다른 모델이며 KV cache도 별도다.

### Q13. Target을 결국 쓰는데 왜 빨라지는가?

Target을 없애는 것이 아니라 `1-token forward × k회`를 `k-token block verification × 1회`로 바꾼다. Memory-bound decode에서 weight/KV 읽기와 kernel-launch 비용을 여러 위치에 분산하고 GPU 병렬성을 이용하기 때문에 Draft가 잘 맞는 범위에서 빨라진다.

### Q14. Acceptance length는 acceptance rate와 무엇이 다른가?

Acceptance length는 Target 검증 한 번에 평균 몇 token을 전진했는지다. `k=3`, acceptance length 3.32는 proposal의 332%가 맞았다는 뜻이 아니라 한 verification step당 correction 또는 bonus를 포함해 평균 3.32 token을 출력했다는 뜻이다.

### Q15. Actual 8B와 simulated 235B의 차이는?

8B는 32개 GB200에서 직접 측정해 generation 1.54-1.77배, synchronous end-to-end 1.35-1.41배를 얻었다. 235B의 약 2.5배는 최대 2048개 GB200 조건을 proprietary simulator로 계산한 전망치다.

### Q16. Weight synchronization은 왜 필요한가?

MegatronLM learner가 Policy를 업데이트해도 vLLM rollout copy는 자동으로 바뀌지 않는다. 최신 weight를 전달하지 않으면 오래된 Policy가 만든 rollout을 새 Policy 기준으로 학습하게 된다. Online Draft도 학습한다면 최신 Draft weight를 rollout worker로 전달해야 한다.

### Q17. Exposed generation time은 무엇인가?

Async에서 generation과 training이 겹친 뒤에도 남아 learner가 rollout을 기다리는 꼬리 시간이다. 이 논문에서는 10.4초에서 0.6초로 감소했다.

---

## 17. 발표자가 암기하면 좋은 숫자 10개

1. Generation 비중: 65-72%
2. RL-Think generation: 133.6초 -> 87.0초
3. RL-Zero generation: 100.0초 -> 56.6초
4. Generation speedup: 1.54배, 1.77배
5. Synchronous end-to-end: 1.35배, 1.41배
6. Async end-to-end: 1.24배
7. Async exposed generation: 10.4초 -> 0.6초
8. 기본 draft length: `k = 3`
9. Actual scale: Qwen3 8B, 32 GB200 GPUs
10. 235B 약 2.5배: actual이 아니라 simulator projection

---

## 18. 30초 요약

이 논문은 reasoning RL에서 가장 비싼 rollout generation을 speculative decoding으로 가속합니다. 작은 EAGLE-3 draft가 여러 token을 제안하고 현재 policy가 verifier로 정확히 검증하기 때문에 최종 rollout distribution은 유지됩니다. 이를 NeMo-RL, vLLM, MegatronLM, GRPO 안에 실제로 통합해 8B synchronous RL에서 generation 1.5-1.8배, 전체 step 1.35-1.41배 속도 향상을 측정했습니다. 핵심 ablation은 in-domain draft와 짧은 `k = 3`이 좋고, online adaptation은 약한 draft에서만 주로 도움이 된다는 것입니다. 235B의 2.5배는 실측이 아니라 simulator projection입니다.

---

## 19. 1분 요약

Reasoning과 agentic RL은 모델이 긴 답을 여러 개 생성하기 때문에 rollout generation이 전체 학습 시간의 대부분을 차지할 수 있습니다. 이 논문에서는 실제 8B workload에서 generation이 step 시간의 65-72%였습니다.

저자들은 NeMo-RL의 vLLM rollout backend에 EAGLE-3 speculative decoding을 통합했습니다. Draft가 여러 token을 먼저 제안하지만, 현재 policy가 verifier가 되어 rejection sampling으로 최종 token distribution을 결정합니다. GRPO의 log-probability와 loss도 draft가 아니라 MegatronLM verifier policy 기준으로 다시 계산합니다.

실제 결과는 generation 1.54-1.77배, synchronous end-to-end 1.35-1.41배, asynchronous 1.24배입니다. DAPO in-domain draft가 generic chat draft보다 좋았고, draft length를 3에서 5나 7로 늘리면 acceptance는 높아져도 실제 speedup은 낮아졌습니다. Online draft adaptation은 초기 draft가 약할 때 도움이 되었지만 잘 초기화된 draft에는 이득이 거의 없었습니다.

따라서 이 논문의 가장 큰 가치는 새로운 decoding 알고리즘보다, verifier-exact speculative decoding을 moving-policy distributed RL system 안에서 실제로 작동하게 만든 데 있습니다.

---

## 20. 3분 요약

이 논문이 해결하려는 문제는 reasoning·agentic RL의 rollout generation 병목입니다. GRPO 같은 학습에서는 하나의 prompt에 대해 여러 답을 길게 생성한 뒤 reward와 상대적 advantage를 계산합니다. 답이 길어질수록 autoregressive decoding의 순차 비용이 커지고, 저자들의 8B baseline에서는 generation이 전체 step 시간의 65-72%를 차지했습니다. 즉 learner의 backward만 빨라져서는 전체 학습이 충분히 빨라지지 않습니다.

저자들의 해법은 speculative decoding을 NeMo-RL 안에 시스템 수준으로 통합하는 것입니다. EAGLE-3 draft가 다음 여러 token을 한꺼번에 제안하고, 현재 RL policy가 verifier로서 그 제안을 병렬 검증합니다. 여기서 draft는 후보만 낼 뿐 최종 답의 분포를 결정하지 않습니다. rejection sampling과 보정 절차 때문에 최종 trajectory는 현재 verifier policy에서 직접 sampling한 것과 같은 분포를 유지합니다. GRPO에 필요한 log-probability와 loss 역시 draft가 아니라 MegatronLM의 verifier policy 기준으로 계산합니다. 따라서 속도를 얻으면서 on-policy 학습의 의미를 보존하는 것이 핵심입니다.

이 통합이 일반 inference보다 어려운 이유는 policy가 계속 변하기 때문입니다. NeMo-RL learner의 policy weight를 vLLM verifier에 동기화해야 하고, draft도 새 policy와 너무 멀어지지 않도록 초기화하거나 online adaptation해야 합니다. 저자들은 synchronous와 asynchronous RL을 모두 지원하고, 일반적인 EAGLE-3 경로와 모델 내부 MTP head를 쓰는 경로도 설계했습니다.

실험은 Qwen3-8B와 Qwen3-8B-Base, DAPO-Math-17K 학습 데이터, AIME-2024 평가, 32개의 GB200 GPU에서 수행했습니다. 실제 synchronous 결과는 generation 1.54-1.77배, 전체 step 1.35-1.41배입니다. 이는 전체 시간이 약 26-29% 줄었다는 뜻입니다. 학습 정확도 곡선은 baseline과 거의 겹쳤습니다. Async 실험은 generation이 이미 training과 겹쳐져 있어 end-to-end 이득이 1.24배로 더 작았습니다.

Ablation이 주는 실무적 교훈도 중요합니다. 첫째, DAPO prompt와 policy response로 만든 in-domain draft가 generic chat 초기화보다 좋았습니다. 둘째, draft length `k`를 3에서 5나 7로 늘리면 acceptance length는 증가하지만 검증 overhead 때문에 실제 speedup은 감소했습니다. RL-Think에서는 `k = 5`와 `k = 7`이 오히려 autoregressive baseline보다 느렸습니다. 셋째, online draft adaptation은 약한 초기 draft에는 도움이 되었지만 좋은 in-domain 초기화에는 거의 추가 이득이 없었습니다. 넷째, 단순 n-gram draft는 acceptance가 2 이상이어도 overhead 때문에 baseline보다 느렸습니다. 따라서 acceptance만 높다고 성공한 것이 아니라 실제 wall-clock을 재야 합니다.

비판적으로 볼 지점도 분명합니다. 235B 모델에서 최대 약 2.5배라는 수치는 실제 학습 측정이 아니라 proprietary simulator projection입니다. 실제 검증은 8B 수학 reasoning workload에 집중되어 있고, FastGRPO나 ReSpec 같은 다른 RL 전용 speculative system과 동일 조건의 직접 비교도 없습니다. 그러므로 이 논문의 결론은 “모든 RL 학습이 2.5배 빨라진다”가 아니라, “rollout 비중이 크고 draft가 policy와 잘 맞으며 system overhead를 통제할 수 있을 때, exact speculative decoding이 RL 학습의 실제 시간을 의미 있게 줄일 수 있다”로 정리해야 합니다.

발표의 마지막에는 이렇게 말하면 좋습니다. 이 연구의 진짜 공헌은 모델이 배우는 답을 바꾼 것이 아니라, 그 답을 기다리는 시간을 줄인 것입니다. 그리고 그 속도 향상은 draft model 하나가 아니라 verifier semantics, weight synchronization, rollout engine, learner를 함께 설계했기 때문에 가능했습니다.

---

## 21. 발표 직전 최종 체크리스트

- [ ] 1.8배를 전체 학습 speedup이라고 말하지 않는다.
- [ ] Actual 8B와 simulated 235B를 구분한다.
- [ ] Draft와 verifier의 역할을 한 문장으로 설명할 수 있다.
- [ ] GRPO log-prob이 verifier policy 기준이라는 점을 설명할 수 있다.
- [ ] Weight synchronization이 왜 필요한지 설명할 수 있다.
- [ ] In-domain initialization 결과를 기억한다.
- [ ] `k`가 클수록 acceptance는 올라가지만 speedup은 내려간 결과를 기억한다.
- [ ] Online adaptation이 항상 유리하지 않았다는 점을 기억한다.
- [ ] Async가 이미 generation을 숨기므로 speedup이 작아진다고 설명한다.
- [ ] Proprietary simulator와 재현성 한계를 언급한다.

- [ ] `Policy = Target = speculative Verifier`이고 reward verifier는 별도라고 설명할 수 있다.
- [ ] Qwen3와 EAGLE-3는 본체와 전용 보조 Draft의 관계라고 설명할 수 있다.
- [ ] Target KV와 Draft KV는 공유하지 않는다고 설명할 수 있다.
- [ ] n-gram proposal과 acceptance length의 뜻을 설명할 수 있다.
- [ ] `k`가 커질 때 proposal·verification·폐기 비용이 왜 증가하는지 설명할 수 있다.
- [ ] Exposed generation time을 “overlap으로 숨기지 못한 대기 꼬리”라고 설명할 수 있다.

---

## 22. 최종 평가

이 논문은 speculative decoding의 새로운 수학적 원리를 제시하는 논문은 아니다. 대신 이미 알려진 lossless decoding 원리를 production-grade RL training stack에 연결하고, 실제 병목과 운영 변수를 계측했다.

가장 설득력 있는 부분은 다음 세 가지다.

1. Generation이 정말 병목임을 stage time으로 증명했다.
2. Verifier-exact semantics와 learner log-prob 경로를 분리했다.
3. Draft initialization, draft length, online adaptation, async interaction을 실무적으로 분석했다.

가장 조심해서 봐야 하는 부분은 다음 세 가지다.

1. Actual experiment가 8B 수학 reasoning에 한정된다.
2. 235B 2.5배는 proprietary simulator projection이다.
3. 다른 RL-specific speculative system과 동일 조건 직접 비교가 없다.

발표의 결론은 다음처럼 잡는 것이 좋다.

> 이 논문은 모델이 배우는 답을 바꾸지 않으면서, 그 답을 만들어 내기 위해 기다리는 시간을 줄이는 방법을 보여 준다. 핵심은 draft model 하나가 아니라, verifier semantics와 distributed RL system 전체를 함께 설계한 것이다.
