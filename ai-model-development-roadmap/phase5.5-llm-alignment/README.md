# Phase 5.5: LLM Alignment & Advanced Architectures

## 🎯 왜 이 Phase가 필요한가?

**"GPT-3는 똑똑하지만, ChatGPT는 유용합니다."**

이 차이를 만드는 것이 바로 **Alignment**입니다!

### ❓ 현실 체크

Phase 2까지 GPT를 학습했다면:
- ✅ 텍스트를 생성할 수 있음
- ❌ 하지만 지시를 따르지 않음
- ❌ 유해한 내용도 생성할 수 있음
- ❌ "도움이 되는" 답변을 선호하지 않음

```python
# Base GPT-3
prompt = "How do I make a cake?"
output = "How do I make a cake? How do I make a pie? ..." # 계속 질문만

# ChatGPT (RLHF 후)
prompt = "How do I make a cake?"
output = "Here's a simple recipe: 1. Preheat oven to 350°F ..." # 유용한 답변!
```

### 🔥 DeepSeek-V3는 왜 빠른가?

**MoE (Mixture of Experts)** 덕분!

```
일반 Transformer: 모든 파라미터를 항상 사용 (느림)
MoE: 필요한 전문가(expert)만 활성화 (빠름!)

DeepSeek-V3: 671B 파라미터, 하지만 37B만 활성화
→ 작은 모델처럼 빠르지만, 큰 모델처럼 똑똑함!
```

---

## 📚 커리큘럼 (2주, 80시간)

### Week 1: LLM Alignment (정렬)

#### [Day 1-2: Instruction Tuning](./01-instruction-tuning.md)
**목표**: Base LM → Instruction-following LM

- **Supervised Fine-Tuning (SFT)**
  - Instruction 데이터셋 (Alpaca, Dolly, ShareGPT)
  - Prompt 템플릿 설계
  - "Below is an instruction..." 포맷
- **데이터셋 구축**
  - Self-Instruct
  - Evol-Instruct
  - 데이터 품질의 중요성
- 💻 **실습**: GPT-2를 instruction-following으로 fine-tune

#### [Day 3-5: RLHF (Reinforcement Learning from Human Feedback)](./02-rlhf.md)
**목표**: ChatGPT의 핵심 기술 완전 이해

**1. Reward Model 훈련**
- 인간 선호도 데이터 수집
  ```
  Prompt: "Python으로 정렬 알고리즘 짜줘"
  Response A: "def sort(arr): ..." ✅ 선호
  Response B: "모르겠어요" ❌ 비선호
  ```
- Pairwise comparison loss
- Bradley-Terry 모델

**2. RL Fine-tuning with PPO**
- **Policy**: LLM (정책 = 텍스트 생성 모델)
- **Reward**: Reward Model의 점수
- **PPO (Proximal Policy Optimization)**
  - Clipped objective
  - KL divergence penalty (원본 모델과 너무 멀어지지 않게)
- Value function 학습

**3. 전체 파이프라인**
```
1. SFT: Base model → Instruction model
2. Reward Model: Human preference 학습
3. PPO: Reward를 최대화하도록 fine-tune
```

💻 **실습**:
- Mini RLHF 구현 (간단한 sentiment로)
- TRL (Transformer Reinforcement Learning) 라이브러리 사용
- ChatGPT 스타일 대화 모델 만들기

#### [Day 6-7: DPO (Direct Preference Optimization)](./03-dpo.md)
**목표**: RLHF보다 간단한 최신 방법

**왜 DPO?**
- RLHF 문제점:
  - 복잡함 (Reward model + PPO)
  - 불안정함 (RL 훈련)
  - 느림 (두 단계)

- DPO 장점:
  - **한 단계로 끝!**
  - 안정적
  - 효율적

**핵심 아이디어**
```
RLHF: Reward model → RL로 maximize
DPO:  직접 preference에서 학습 (no Reward model!)
```

**수식**
$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

💻 **실습**:
- DPO 직접 구현
- RLHF vs DPO 성능 비교

---

### Week 2: Advanced Architectures

#### [Day 8-10: MoE (Mixture of Experts)](./04-mixture-of-experts.md)
**목표**: DeepSeek-V3, Mixtral의 아키텍처 이해

**MoE 핵심 개념**

```python
# 일반 FFN
def ffn(x):
    return W2 @ relu(W1 @ x)  # 모든 파라미터 사용

# MoE
def moe(x):
    # 1. Router가 어떤 expert 사용할지 결정
    router_logits = router(x)  # (batch, num_experts)
    top_k_indices = topk(router_logits, k=2)  # 2개 expert만 선택

    # 2. 선택된 expert만 실행
    output = 0
    for idx in top_k_indices:
        output += experts[idx](x) * router_weights[idx]

    return output
```

**왜 효율적?**
- 전체 파라미터: 100B
- 활성화: 13B (8개 중 2개 expert)
- **계산량은 13B, 용량은 100B!**

**구현 요소**
1. **Router (Gating) Network**
   - Softmax routing
   - Top-K selection
   - Load balancing (expert들이 골고루 사용되게)

2. **Expert Networks**
   - 일반적으로 FFN들
   - 각각 독립적으로 학습

3. **Training Challenges**
   - Load balancing loss
   - Expert 붕괴 방지
   - Routing stability

**실전 모델들**
- **Switch Transformer** (Google, 2021)
  - 1.6T 파라미터
  - 각 토큰이 1개 expert만 사용

- **Mixtral 8x7B** (Mistral AI, 2023)
  - 8개 expert, Top-2 routing
  - 46.7B 파라미터, 12.9B 활성화

- **DeepSeek-V3** (DeepSeek, 2024)
  - 671B 파라미터, 37B 활성화
  - Multi-head latent attention
  - Load balancing 개선

💻 **실습**:
- 간단한 MoE layer 구현
- Load balancing 시각화
- DeepSeek-V3 스타일 MoE 구현

#### [Day 11-12: Constitutional AI & RLAIF](./05-constitutional-ai.md)
**목표**: Anthropic (Claude)의 alignment 방법

**Constitutional AI**

인간 피드백 없이 AI가 AI를 정렬!

```
1. Red-teaming: AI가 유해한 출력 생성
2. Self-critique: AI가 자신의 출력 비판
3. Revision: AI가 개선된 출력 생성
4. RL from AI Feedback (RLAIF)
```

**Constitution (헌법)**
```
원칙들:
- "Be helpful, harmless, and honest"
- "Avoid stereotypes"
- "Don't help with illegal activities"
- ...
```

💻 **실습**:
- Self-critique prompt 설계
- RLAIF vs RLHF 비교

#### [Day 13-14: 종합 실습](./06-putting-it-together.md)

**프로젝트: Mini ChatGPT 구축**

```
단계별 구현:
1. Base model (GPT-2)
2. Instruction tuning (Alpaca 데이터)
3. Reward model 훈련
4. PPO fine-tuning
5. 또는 DPO로 대체
6. 평가 및 비교
```

**평가 방법**
- MT-Bench
- AlpacaEval
- Human evaluation

---

## 🎓 학습 목표

### 이론 이해
- [ ] RLHF의 3단계 파이프라인 설명 가능
- [ ] PPO의 clipped objective 이해
- [ ] DPO가 왜 simpler한지 설명 가능
- [ ] MoE의 routing 메커니즘 이해
- [ ] Load balancing의 중요성 이해

### 실습 완료
- [ ] GPT-2에 instruction tuning 적용
- [ ] Reward model 직접 훈련
- [ ] PPO로 mini RLHF 구현
- [ ] DPO 구현 및 비교
- [ ] MoE layer 구현
- [ ] Mini ChatGPT 완성

### AI 연결
- [ ] ChatGPT vs GPT-3 차이 설명 가능
- [ ] Claude의 Constitutional AI 원리 이해
- [ ] DeepSeek-V3의 효율성 비밀 이해
- [ ] Mixtral의 MoE 아키텍처 분석 가능

---

## 📊 Before & After

### Before (이 Phase 없이)
```
"ChatGPT가 뭔가요?"
→ "GPT인데... 좀 더 좋은 거?"

"DeepSeek-V3는 왜 빠른가요?"
→ "글쎄요... 최적화?"

"어떻게 유해 콘텐츠를 막나요?"
→ "필터링?"
```

### After (이 Phase 완료 후)
```
"ChatGPT가 뭔가요?"
→ "GPT-3.5에 Instruction Tuning + RLHF를 적용한 모델입니다.
   인간 선호도를 학습한 Reward Model로 PPO를 통해 정렬했죠."

"DeepSeek-V3는 왜 빠른가요?"
→ "671B 파라미터의 MoE 모델인데, Top-K routing으로
   37B만 활성화하기 때문입니다. Sparse activation이 핵심이죠."

"어떻게 유해 콘텐츠를 막나요?"
→ "RLHF의 Reward Model이 유해성을 낮게 점수 매기고,
   PPO가 그 방향으로 학습합니다. 또는 Constitutional AI로
   AI가 스스로 비판하고 수정하게 할 수도 있습니다."
```

---

## 🔥 실전 응용

### 1. 자신만의 ChatGPT 만들기
```python
from transformers import AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# 1. Base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 2. Instruction tuning
# (alpaca dataset으로 SFT)

# 3. RLHF
ppo_trainer = PPOTrainer(
    model=model,
    reward_model=reward_model,
    config=ppo_config
)
ppo_trainer.train()
```

### 2. 효율적인 대형 모델 설계
```python
# MoE로 효율성 극대화
class MoETransformer(nn.Module):
    def __init__(self, num_experts=8, expert_capacity=2):
        self.experts = nn.ModuleList([
            FFN() for _ in range(num_experts)
        ])
        self.router = Router()

    def forward(self, x):
        # Top-K routing
        expert_indices = self.router(x, k=2)
        output = self.moe_forward(x, expert_indices)
        return output
```

### 3. Alignment 평가
```python
# MT-Bench, AlpacaEval로 모델 평가
from alpaca_eval import evaluate

results = evaluate(
    model=your_model,
    reference_model="gpt-4",
    dataset="alpaca_eval"
)
```

---

## 📚 필수 논문

### RLHF
- **InstructGPT** (OpenAI, 2022) - RLHF의 바이블
- **Learning to summarize from human feedback** (OpenAI, 2020)

### DPO
- **Direct Preference Optimization** (Rafailov et al., 2023)
- **RRHF** (Yuan et al., 2023)

### MoE
- **Switch Transformers** (Fedus et al., 2021)
- **Mixtral of Experts** (Mistral AI, 2023)
- **DeepSeek-V3** (DeepSeek, 2024)

### Constitutional AI
- **Constitutional AI** (Anthropic, 2022)
- **RLAIF** (Lee et al., 2023)

---

## 🎯 완료 기준

이 Phase를 완료하면:

- ✅ **ChatGPT를 직접 만들 수 있음**
- ✅ **최신 SOTA 모델들의 핵심 기술 이해**
- ✅ **LLM을 사용자 선호에 맞게 정렬 가능**
- ✅ **효율적인 대형 모델 설계 가능**
- ✅ **논문: DeepSeek-V3, Mixtral, GPT-4 기술 보고서 이해**

---

## ⏭️ 다음 단계

Phase 5.5를 완료했다면, 진정한 현대 LLM 전문가입니다!

👉 [Phase 6: 실전 프로젝트](../phase6-project/)로 모든 지식을 종합하세요!

**"이제 당신은 최신 AI의 모든 것을 이해합니다!"** 🚀
