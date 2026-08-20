# Pretraining Large Language Models with NVFP4

대학생 2학년 수준의 청중도 따라올 수 있도록, NVFP4 기초부터 논문의 training methodology와 실험 결과까지 단계적으로 설명하는 논문 리뷰 자료입니다.

## 발표 자료

- [PowerPoint](./pretraining-llms-with-nvfp4-paper-review.pptx)
- 논문: [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)

## 핵심 흐름

1. 양자화, scale, block scaling, FP4의 기본 개념
2. NVFP4와 MXFP4의 형식 차이
3. 12B hybrid model을 10T tokens로 학습한 실험 설정
4. naive FP4 training이 발산하는 이유
5. 논문의 네 가지 해결책
   - 민감한 layer를 BF16으로 보호
   - Wgrad 입력에 16×16 RHT 적용
   - weight에 16×16 2D scaling 적용
   - gradient operand에 stochastic rounding 적용
6. FP8 대비 loss·downstream accuracy와 ablation 결과
7. 논문의 한계와 발표 예상 질문 10개

## 해석 시 주의점

- 이 실험은 모델 전체를 FP4로 저장하거나 계산한 full-stack FP4 training이 아닙니다.
- NVFP4는 주로 linear GEMM의 입력 operand에 적용되며, output과 optimizer state 등은 더 높은 정밀도를 사용합니다.
- 논문의 Tensor Core 처리량 배수는 end-to-end 학습 속도 향상을 직접 의미하지 않습니다.
- 주요 downstream 평가는 학습 완료 checkpoint를 BF16으로 평가한 결과입니다.
