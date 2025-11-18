# 최신 AI 연구 트렌드 (2024-2025)

## 🎯 목표

**2024-2025년 AI 연구의 최전선: Agent-to-Agent를 넘어서**

최신 연구 트렌드는 단순한 multi-agent를 넘어 다음 영역으로 확장:
- Test-Time Compute Scaling (추론 시간 확장)
- World Models (환경 시뮬레이션)
- Constitutional AI (AI 자체 안전성)
- Diffusion Models 진화
- Efficient Architectures (Mamba, RWKV)

---

## 🔥 Top 10 Research Trends (2024-2025)

### 1. **Test-Time Compute Scaling** ⭐⭐⭐⭐⭐

**혁명적 발견**: 추론 시간에 더 많이 "생각"하면 성능 향상!

```
기존 패러다임:
- Scaling Law: 모델 크기 ↑ = 성능 ↑
- GPT-3 (175B) → GPT-4 (?B)

새로운 패러다임 (OpenAI o1, o3):
- 추론 시간에 더 오래 "생각"
- Chain-of-Thought를 내부적으로 수행
- 작은 모델도 긴 추론으로 큰 모델 능가 가능!

성능:
┌──────────────┬─────────────┬───────────────┐
│   Model      │  Params     │  Reasoning    │
├──────────────┼─────────────┼───────────────┤
│ GPT-4 Turbo  │ ?B          │ 즉각적        │
│ o1-preview   │ ?B (추정소)│ ~10-60초      │
│ o1-mini      │ ?B (더 소)  │ ~5-30초       │
│ o3-mini      │ ?B          │ Adjustable    │
└──────────────┴─────────────┴───────────────┘

벤치마크 (AIME 2024 - 수학 경시대회):
- GPT-4: 13.4%
- o1: 83.3%
- o3: 96.7% (추정)

핵심 아이디어:
"Think slow, answer well"
```

**기술 구현** (추정):

```python
class TestTimeComputeScaling:
    """
    Test-Time Compute Scaling 개념 구현

    핵심: 추론 시 여러 경로 탐색 → 최선 선택
    """

    def __init__(self, base_model, num_thoughts=5):
        self.model = base_model
        self.num_thoughts = num_thoughts

    def generate_with_thinking(self, prompt):
        """
        긴 추론 과정

        Process:
        1. 여러 사고 경로 생성 (Tree of Thoughts)
        2. 각 경로 평가
        3. 최선 경로 선택
        4. 최종 답변 생성

        의도: 단순 다음 토큰 예측 → 계획적 추론
        """

        # Step 1: 내부 사고 과정 생성
        thoughts = []
        for _ in range(self.num_thoughts):
            # 다양한 추론 경로 시도
            thought = self.model.generate(
                f"{prompt}\n\nLet me think step by step:"
            )
            thoughts.append(thought)

        # Step 2: 각 사고 경로 평가
        scores = []
        for thought in thoughts:
            # Self-evaluation
            score = self.evaluate_thought(thought)
            scores.append(score)

        # Step 3: 최선 경로 선택
        best_thought = thoughts[np.argmax(scores)]

        # Step 4: 최종 답변
        final_answer = self.model.generate(
            f"{prompt}\n\nThought process:\n{best_thought}\n\nFinal answer:"
        )

        return final_answer, best_thought

    def evaluate_thought(self, thought):
        """
        사고 경로 평가

        방법:
        - Process reward model (각 단계 평가)
        - Outcome reward model (최종 결과 평가)
        - Self-consistency (여러 경로 일치도)
        """
        # 구현 세부사항은 비공개
        return self.reward_model.score(thought)


"""
응용:
- 수학 문제: 여러 풀이 시도 → 검증
- 코딩: 여러 구현 → 테스트
- 복잡한 추론: 다양한 논리 → 검증

트레이드오프:
- Latency ↑ (10-60초)
- Cost ↑ (더 많은 토큰 생성)
- Accuracy ↑↑ (극적 향상)

사용처:
- 정답이 중요한 경우 (수학, 코딩, 과학)
- Latency 허용 가능 (연구, 분석)
"""
```

**Tree of Thoughts (ToT)**:

```python
class TreeOfThoughts:
    """
    Tree of Thoughts

    논문: "Tree of Thoughts: Deliberate Problem Solving
           with Large Language Models" (Yao et al., 2023)

    핵심: BFS/DFS로 사고 공간 탐색
    """

    def solve(self, problem, depth=3, breadth=3):
        """
        ToT로 문제 해결

        Args:
            depth: 탐색 깊이
            breadth: 각 단계에서 시도할 경로 수

        의도: 체스처럼 여러 수 앞을 내다봄
        """

        # Root: 초기 상태
        root = ThoughtNode(problem, state=None)

        # BFS 탐색
        queue = [root]

        for level in range(depth):
            next_queue = []

            for node in queue:
                # 각 노드에서 breadth개 child 생성
                children = self.generate_children(node, k=breadth)

                # 각 child 평가
                for child in children:
                    child.value = self.evaluate_state(child.state)

                # Top-k 선택
                top_children = sorted(children, key=lambda x: x.value, reverse=True)[:breadth]
                next_queue.extend(top_children)

            queue = next_queue

        # 최선 경로 선택
        best_node = max(queue, key=lambda x: x.value)

        return self.extract_solution(best_node)

    def generate_children(self, node, k=3):
        """
        k개의 다음 단계 생성

        의도: 다양한 접근 시도
        """
        prompt = f"""
Problem: {node.problem}
Current state: {node.state}

Generate {k} different next steps to solve this problem.
"""
        # LLM으로 k개 생성
        children = []
        for i in range(k):
            next_step = self.llm.generate(prompt, temperature=0.8)
            child = ThoughtNode(node.problem, state=next_step, parent=node)
            children.append(child)

        return children

    def evaluate_state(self, state):
        """
        상태 평가

        방법:
        1. LLM self-evaluation
        2. Heuristic (도메인 지식)
        3. Reward model
        """
        prompt = f"""
Rate the following reasoning step on a scale of 1-10:
{state}

Score (1-10):"""

        score = self.llm.generate(prompt, max_tokens=2)
        return float(score)


"""
실제 예제: 24 Game

문제: 4개 숫자로 24 만들기
입력: [4, 9, 10, 13]

ToT 탐색:
Level 0: [4, 9, 10, 13]
Level 1:
  - (13-9) * (10-4) = 4 * 6 = 24 ✓
  - (13-10) * 9 + 4 = 27 ✗
  - ...

최선 경로 선택 → 답 도출
"""
```

---

### 2. **World Models** ⭐⭐⭐⭐⭐

**개념**: 환경을 시뮬레이션하는 생성 모델

```
World Model이란?
- 환경의 동역학을 학습
- 행동의 결과 예측
- 가상으로 미래 시뮬레이션

응용:
- 게임 AI
- 로봇 시뮬레이션
- 자율주행 시뮬레이션
- 비디오 생성

대표 연구:
1. Genie (Google DeepMind, 2024)
   - 비디오만으로 플레이 가능한 게임 생성
   - 11B 파라미터

2. GameNGen (Google, 2024)
   - DOOM을 neural network로 실시간 생성
   - 20+ FPS

3. DIAMOND (Meta, 2024)
   - Diffusion World Model
   - 오프라인 RL
```

**Genie 개념**:

```python
class Genie:
    """
    Genie: Generative Interactive Environments

    논문: "Genie: Generative Interactive Environments" (DeepMind, 2024)

    핵심:
    - 입력: 비디오 (레이블 없음)
    - 출력: 플레이 가능한 환경
    - 행동 공간 자동 학습
    """

    def __init__(self):
        self.video_tokenizer = VideoTokenizer()  # 비디오 → discrete tokens
        self.dynamics_model = DynamicsModel()    # (s_t, a_t) → s_{t+1}
        self.latent_action_model = LatentActionModel()  # 잠재 행동 학습

    def train(self, videos):
        """
        비디오만으로 훈련

        Process:
        1. 비디오 → latent states
        2. 연속 frame 간 latent action 추론
        3. Dynamics model 학습: (s_t, a_t) → s_{t+1}

        의도: 명시적 action label 없이 학습!
        """

        for video in videos:
            # 1. Tokenize frames
            frames = self.video_tokenizer.encode(video)  # (T, h, w, c) → (T, d)

            # 2. 잠재 행동 추론
            # 의도: frame t → frame t+1로 가는 "행동" 추론
            latent_actions = []
            for t in range(len(frames) - 1):
                # frame 차이로부터 action 추론
                action = self.latent_action_model.infer(frames[t], frames[t+1])
                latent_actions.append(action)

            # 3. Dynamics model 훈련
            for t in range(len(frames) - 1):
                s_t = frames[t]
                a_t = latent_actions[t]
                s_next_pred = self.dynamics_model(s_t, a_t)

                # Loss: 예측 frame vs 실제 frame
                loss = mse_loss(s_next_pred, frames[t+1])
                loss.backward()

    def play(self, initial_frame, actions):
        """
        환경과 상호작용

        Args:
            initial_frame: 시작 화면
            actions: 사용자 입력

        Returns:
            생성된 비디오 (게임 플레이)
        """
        current_state = self.video_tokenizer.encode(initial_frame)
        generated_frames = [initial_frame]

        for action in actions:
            # 다음 frame 예측
            next_state = self.dynamics_model(current_state, action)

            # Decode to pixel space
            next_frame = self.video_tokenizer.decode(next_state)
            generated_frames.append(next_frame)

            current_state = next_state

        return generated_frames


"""
Genie의 놀라운 점:

1. 행동 라벨 불필요
   - YouTube 게임 영상만으로 학습
   - Latent action space 자동 발견

2. 제로샷 제어
   - 새로운 이미지 (예: 그림)를 주면
   - 그 스타일로 플레이 가능한 게임 생성!

3. 확장성
   - 11B 파라미터
   - 수백만 개 비디오로 훈련

응용:
- 프로토타입 게임 생성
- 가상 환경 생성 (RL 훈련용)
- 교육 시뮬레이션
"""
```

**GameNGen - Neural Game Engine**:

```python
class GameNGen:
    """
    GameNGen: Neural Game Engine

    논문: "Diffusion Models Are Real-Time Game Engines" (Google, 2024)

    혁신: DOOM을 neural network로 실시간 생성!
    - 20.3 FPS
    - PSNR 29.4 dB (고품질)
    """

    def __init__(self):
        self.diffusion_model = DiffusionModel()

    def train_phase1_rl(self):
        """
        Phase 1: RL agent로 게임 플레이 데이터 수집

        의도: 다양한 게임 상황 커버
        """
        env = DoomEnv()
        agent = PPOAgent()

        # RL agent 훈련
        for episode in range(10000):
            states, actions, rewards = agent.play_episode(env)
            agent.update(states, actions, rewards)

        # 데이터 저장
        training_data = collect_trajectories(agent, env, num_episodes=1000000)
        return training_data

    def train_phase2_diffusion(self, training_data):
        """
        Phase 2: Diffusion model 훈련

        입력: (frames_{t-3:t}, action_t)
        출력: frame_{t+1}

        의도: 다음 frame 생성
        """
        for batch in training_data:
            # 이전 4 frames + action → 다음 frame
            past_frames = batch['frames'][-4:]  # (4, H, W, 3)
            action = batch['action']
            next_frame_true = batch['next_frame']

            # Condition: past frames + action
            condition = concatenate([past_frames, action])

            # Diffusion training
            noise = randn_like(next_frame_true)
            t = randint(0, self.num_timesteps)

            # Forward diffusion
            noisy_frame = self.add_noise(next_frame_true, noise, t)

            # Predict noise
            noise_pred = self.diffusion_model(noisy_frame, t, condition)

            # Loss
            loss = mse_loss(noise_pred, noise)
            loss.backward()

    def play_realtime(self, user_actions):
        """
        실시간 게임 플레이

        의도: Diffusion sampling을 충분히 빠르게
        """
        frame_buffer = initialize_frames()

        for action in user_actions:
            # Condition
            condition = concatenate([frame_buffer[-4:], action])

            # Fast sampling (single-step)
            # 의도: 20+ FPS 달성
            next_frame = self.diffusion_model.fast_sample(
                condition,
                num_steps=1  # DDIM single-step!
            )

            frame_buffer.append(next_frame)
            render(next_frame)


"""
GameNGen 성능:

PSNR (Peak Signal-to-Noise Ratio):
- GameNGen: 29.4 dB
- JPEG (lossy compression): ~30 dB
- 거의 구분 불가능!

Latency:
- 50ms per frame (20 FPS)
- 실시간 플레이 가능

한계:
- 장기 일관성 (long-term consistency)
- 복잡한 게임 로직 (예: 인벤토리)

미래:
- 더 복잡한 게임
- 유저가 만드는 게임 엔진
"""
```

---

### 3. **Diffusion Models 진화** ⭐⭐⭐⭐⭐

**최신 발전**: 이미지 → 비디오 → 3D → 게임

```
Diffusion Models Timeline:

2020: DDPM (Denoising Diffusion Probabilistic Models)
2021: Improved DDPM, Guided Diffusion
2022: DALL-E 2, Stable Diffusion 1.x
2023: Stable Diffusion XL, Midjourney v5
2024: Stable Diffusion 3, Sora, Emu Video

핵심 발전:
1. 해상도 ↑ (512 → 1024 → 2048)
2. 속도 ↑ (50 steps → 4 steps)
3. 제어성 ↑ (text → text+image+pose)
4. 차원 ↑ (2D → Video → 3D)
```

**Sora - Text-to-Video**:

```python
class Sora:
    """
    Sora: OpenAI의 Text-to-Video 모델 (2024)

    특징:
    - 최대 1분 비디오
    - 1920x1080 해상도
    - 물리 법칙 이해
    - Temporal consistency

    기술 (추정):
    - Diffusion Transformer (DiT)
    - 3D convolution + Temporal attention
    - Latent space diffusion
    """

    def __init__(self):
        self.video_vae = VideoVAE()  # 비디오 압축
        self.transformer = DiffusionTransformer()  # Denoising
        self.text_encoder = T5Encoder()  # Text conditioning

    def generate_video(self, text_prompt, num_frames=240, resolution=(1920, 1080)):
        """
        텍스트로부터 비디오 생성

        Args:
            text_prompt: "A cat walking on the moon"
            num_frames: 240 frames (60 FPS × 4초)
            resolution: (1920, 1080)

        Process:
        1. Text → embedding
        2. Random noise (latent space)
        3. Iterative denoising
        4. VAE decode → pixel space
        """

        # 1. Text encoding
        text_embedding = self.text_encoder(text_prompt)

        # 2. Random latent (압축된 공간)
        # 의도: Pixel space (1920x1080x240)는 너무 큼
        #       Latent space (120x68x60)로 압축
        latent_shape = (num_frames // 4, resolution[0] // 16, resolution[1] // 16, 4)
        z_T = randn(latent_shape)  # Pure noise

        # 3. Diffusion sampling
        z_0 = self.denoise(z_T, text_embedding, num_steps=50)

        # 4. Decode
        video = self.video_vae.decode(z_0)  # (240, 1920, 1080, 3)

        return video

    def denoise(self, z_T, text_embedding, num_steps=50):
        """
        Iterative denoising

        의도: Noise → Video
        """
        z_t = z_T

        for t in reversed(range(num_steps)):
            # Predict noise
            noise_pred = self.transformer(
                z_t,
                timestep=t,
                condition=text_embedding
            )

            # DDIM update
            z_t = self.ddim_step(z_t, noise_pred, t)

        return z_t  # z_0 (clean video)


"""
Sora의 능력:

1. 물리 이해
   - 중력, 관성, 충돌
   - 물체 영속성 (object permanence)

2. Temporal consistency
   - 프레임 간 일관성
   - 카메라 움직임

3. 복잡한 장면
   - 여러 캐릭터
   - 배경 + 전경
   - 조명 변화

4. 긴 비디오
   - 최대 60초
   - 일관된 내러티브

한계:
- 복잡한 물리 (예: 유체)
- 텍스트 렌더링
- 사람 손가락 (여전히!)
- Hallucination (환각)

비용:
- 1분 비디오 생성: ~$10-20 (추정)
- 생성 시간: 수 분
"""
```

**Stable Diffusion 3**:

```python
class StableDiffusion3:
    """
    Stable Diffusion 3 (Stability AI, 2024)

    혁신:
    - Multimodal Diffusion Transformer (MMDiT)
    - Flow matching (vs DDPM)
    - 3개 text encoder (CLIP + T5)

    성능:
    - 텍스트 렌더링 개선
    - Prompt adherence 향상
    - 8B 파라미터
    """

    def __init__(self):
        # 3개 text encoder
        self.clip_l = CLIPTextModel()
        self.clip_g = CLIPTextModelBig()
        self.t5 = T5Encoder()

        # MMDiT
        self.mmdit = MMDiT(depth=24, width=1536)

        # VAE
        self.vae = AutoencoderKL()

    def encode_prompt(self, text):
        """
        3개 encoder로 텍스트 인코딩

        의도: 다양한 수준의 이해
        - CLIP-L: 짧은 개념
        - CLIP-G: 글로벌 의미
        - T5: 긴 문장, 문법
        """
        emb_clip_l = self.clip_l(text)
        emb_clip_g = self.clip_g(text)
        emb_t5 = self.t5(text)

        # Concatenate
        combined = torch.cat([emb_clip_l, emb_clip_g, emb_t5], dim=-1)

        return combined

    def mmdit_block(self, img_latent, text_latent):
        """
        Multimodal DiT Block

        핵심: Image와 Text latent를 함께 처리

        구조:
        img_latent → Self-Attention → Cross-Attention ← text_latent
                  ↓                                    ↓
                 FFN                                  FFN
                  ↓                                    ↓
               img_out                              text_out
        """
        # Image self-attention
        img_attn = self.self_attention(img_latent)

        # Cross-attention (img ← text)
        img_cross = self.cross_attention(img_attn, text_latent)

        # FFN
        img_out = self.ffn(img_cross + img_latent)

        # Text도 동일 과정
        text_attn = self.self_attention(text_latent)
        text_cross = self.cross_attention(text_attn, img_latent)
        text_out = self.ffn(text_cross + text_latent)

        return img_out, text_out


"""
SD3 vs SD2.1:

텍스트 렌더링:
- SD2.1: "HELLO" → "HLLO" or "HELO" (오타)
- SD3: "HELLO" → "HELLO" (정확!)

Prompt Adherence:
- SD2.1: "red car, blue house" → 종종 색 혼동
- SD3: 정확한 색상, 위치, 수량

속도:
- SD3: 28 steps (vs SD2.1 50 steps)
- Distilled version: 4 steps!

오픈소스:
- 2024년 6월 공개
- 상업적 사용 가능 (라이선스 확인)
"""
```

---

### 4. **Efficient Architectures Beyond Transformers** ⭐⭐⭐⭐⭐

**동기**: Transformer의 O(n²) 복잡도 해결

```
Transformer 한계:
- Attention: O(n²) 시간, 메모리
- Long context: 100K tokens → 10B operations!

대안 아키텍처:
1. Mamba (State Space Models)
2. RWKV (Receptance Weighted Key Value)
3. Hyena (Long convolution)
4. RetNet (Retentive Networks)

공통 목표:
- O(n) 복잡도
- Long context 효율
- 성능 유지/향상
```

**Mamba - State Space Models**:

```python
class Mamba:
    """
    Mamba: Linear-Time Sequence Modeling

    논문: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)

    핵심:
    - Selective SSM (State Space Model)
    - O(n) 복잡도
    - 100만+ token 처리 가능

    성능:
    - Transformer와 동등 또는 우수
    - 5배 빠른 추론
    """

    def __init__(self, d_model=768, d_state=16):
        """
        Args:
            d_model: 모델 차원
            d_state: State 차원 (작음!)
        """
        self.d_model = d_model
        self.d_state = d_state

        # SSM 파라미터
        self.A = nn.Parameter(torch.randn(d_model, d_state))  # State transition
        self.B = nn.Parameter(torch.randn(d_model, d_state))  # Input projection
        self.C = nn.Parameter(torch.randn(d_model, d_state))  # Output projection
        self.D = nn.Parameter(torch.randn(d_model))  # Skip connection

        # Selective mechanism
        self.delta_proj = nn.Linear(d_model, d_model)  # Time-varying

    def forward(self, x):
        """
        Selective SSM Forward

        State Space Model:
        h_t = A·h_{t-1} + B·x_t
        y_t = C·h_t + D·x_t

        Selective:
        A, B, C가 입력에 따라 변함!

        의도: 중요한 정보 선택적 기억
        """
        batch, seq_len, d = x.shape

        # Selective parameters
        # 의도: 입력에 따라 A, B, C 조정
        delta = F.softplus(self.delta_proj(x))  # (batch, seq, d)

        # SSM computation (simplified)
        h = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, d)

            # Selective update
            # 의도: delta로 update 속도 조절
            A_t = self.A * delta[:, t:t+1, :]
            B_t = self.B * delta[:, t:t+1, :]

            # State update
            h = A_t @ h + B_t @ x_t.unsqueeze(-1)

            # Output
            y_t = (self.C @ h).squeeze(-1) + self.D * x_t

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


"""
Mamba의 장점:

1. 복잡도
   - Attention: O(n²)
   - Mamba: O(n)

2. 추론 속도
   - Transformer: n개 token → n² operations
   - Mamba: n개 token → n operations
   - 5배 빠름!

3. Long context
   - 1M token도 처리 가능
   - 메모리 효율적

4. 성능
   - Language modeling: Transformer와 동등
   - Long-range tasks: Transformer 능가

응용:
- 긴 문서 처리
- DNA 서열 분석 (백만+ bp)
- Audio generation (hours)
- Time series (millions of points)

한계:
- 병렬화 어려움 (sequential)
- 생태계 작음 (Transformer 대비)
"""
```

**RWKV - Receptance Weighted Key Value**:

```python
class RWKV:
    """
    RWKV: Reinventing RNNs for the Transformer Era

    논문: "RWKV: Reinventing RNNs for the Transformer Era" (Peng et al., 2023)

    특징:
    - RNN처럼 효율적
    - Transformer처럼 병렬 훈련 가능
    - O(n) 복잡도
    """

    def __init__(self, d_model=768):
        self.d_model = d_model

        # Learnable parameters
        self.time_mix_k = nn.Parameter(torch.randn(d_model))
        self.time_mix_v = nn.Parameter(torch.randn(d_model))
        self.time_mix_r = nn.Parameter(torch.randn(d_model))

        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.receptance = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x, state=None):
        """
        RWKV Forward

        핵심 아이디어:
        - Time-mixing: 이전 상태와 현재 입력 혼합
        - Linear attention: O(n) 복잡도

        의도: RNN의 효율성 + Transformer의 성능
        """
        batch, seq_len, d = x.shape

        if state is None:
            state = torch.zeros(batch, d, device=x.device)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            # Time mixing
            # 의도: 이전 상태와 현재 입력 가중 평균
            k_input = self.time_mix_k * state + (1 - self.time_mix_k) * x_t
            v_input = self.time_mix_v * state + (1 - self.time_mix_v) * x_t
            r_input = self.time_mix_r * state + (1 - self.time_mix_r) * x_t

            # K, V, R 계산
            k = self.key(k_input)
            v = self.value(v_input)
            r = torch.sigmoid(self.receptance(r_input))

            # WKV (Weighted Key Value)
            # 의도: Attention과 유사하지만 O(n)
            wkv = k * v  # Element-wise (vs Attention의 QK^T)

            # Output
            y_t = r * wkv
            y_t = self.output(y_t)

            outputs.append(y_t)

            # State update
            state = x_t  # Simplified

        return torch.stack(outputs, dim=1), state


"""
RWKV 특징:

1. 훈련: 병렬 가능
   - Transformer처럼 전체 시퀀스 한 번에
   - GPU 효율적

2. 추론: Sequential
   - RNN처럼 state 유지
   - Constant memory
   - 빠른 생성 속도

3. 성능:
   - Language modeling: GPT와 유사
   - Long context: 우수

모델 크기:
- RWKV-4: 169M, 1.5B, 3B, 7B, 14B
- RWKV-5: 더 개선된 버전

사용처:
- 채팅 봇 (긴 대화)
- 실시간 생성 (낮은 latency)
- 리소스 제약 환경
"""
```

---

### 5. **Constitutional AI / RLAIF** ⭐⭐⭐⭐

**개념**: AI가 AI를 안전하게 만들기

```
기존: RLHF (Reinforcement Learning from Human Feedback)
- 인간이 선호도 라벨링
- 비용 ↑, 확장성 ↓

새로운: RLAIF (RL from AI Feedback)
- AI가 AI 평가
- 비용 ↓, 확장성 ↑

Constitutional AI (Anthropic):
- AI에게 "헌법" 제공
- AI가 스스로 안전성 평가
```

**Constitutional AI 구현**:

```python
class ConstitutionalAI:
    """
    Constitutional AI (Anthropic, 2022)

    논문: "Constitutional AI: Harmlessness from AI Feedback"

    Process:
    1. Supervised learning
    2. AI feedback (critique + revision)
    3. RL from AI feedback
    """

    def __init__(self, base_model, constitution):
        """
        Args:
            base_model: Base LLM
            constitution: 안전성 원칙 리스트
        """
        self.model = base_model
        self.constitution = constitution

    def critique_and_revise(self, prompt, response):
        """
        Phase 1: Critique and Revision

        Process:
        1. 초기 응답 생성
        2. Constitution 기반 비평
        3. 개선된 응답 생성

        의도: Self-improvement loop
        """

        # 1. 초기 응답
        initial_response = self.model.generate(prompt)

        # 2. 각 원칙에 대해 비평
        critiques = []
        for principle in self.constitution:
            critique_prompt = f"""
Principle: {principle}

Prompt: {prompt}
Response: {initial_response}

Does this response violate the principle? If so, how?

Critique:"""

            critique = self.model.generate(critique_prompt)
            critiques.append(critique)

        # 3. 비평 기반 수정
        revision_prompt = f"""
Original prompt: {prompt}
Original response: {initial_response}

Critiques:
{chr(10).join(critiques)}

Please revise the response to address these critiques while still being helpful.

Revised response:"""

        revised_response = self.model.generate(revision_prompt)

        return revised_response

    def train_with_ai_feedback(self, prompts):
        """
        Phase 2: RL from AI Feedback

        Process:
        1. 여러 응답 생성
        2. AI가 응답 평가 (constitution 기반)
        3. Preference pairs 생성
        4. Reward model 훈련
        5. PPO로 policy 최적화

        의도: 인간 피드백 없이 RLHF
        """

        preference_data = []

        for prompt in prompts:
            # 여러 응답 생성
            responses = [
                self.model.generate(prompt, temperature=0.8)
                for _ in range(4)
            ]

            # AI가 평가
            scores = []
            for response in responses:
                score = self.evaluate_response(prompt, response)
                scores.append(score)

            # Preference pairs
            # 의도: 더 좋은 응답 vs 나쁜 응답
            best_idx = np.argmax(scores)
            worst_idx = np.argmin(scores)

            preference_data.append({
                'prompt': prompt,
                'chosen': responses[best_idx],
                'rejected': responses[worst_idx]
            })

        # Reward model 훈련
        reward_model = self.train_reward_model(preference_data)

        # PPO
        self.model = self.ppo_train(self.model, reward_model)

        return self.model

    def evaluate_response(self, prompt, response):
        """
        AI가 응답 평가

        의도: Constitution 준수 정도
        """
        score = 0

        for principle in self.constitution:
            eval_prompt = f"""
Principle: {principle}

Prompt: {prompt}
Response: {response}

Does this response follow the principle?
Answer with a score from 0 (completely violates) to 10 (perfectly follows).

Score:"""

            score_str = self.model.generate(eval_prompt, max_tokens=2)
            score += float(score_str) / len(self.constitution)

        return score


# Constitution 예시
constitution = [
    "Choose the response that is most helpful, honest, and harmless.",
    "Choose the response that is least racist, sexist, or otherwise discriminatory.",
    "Choose the response that does not encourage illegal, unethical, or immoral activity.",
    "Choose the response that is most thoughtful and considerate.",
]

# 사용
cai = ConstitutionalAI(base_model, constitution)
safe_model = cai.train_with_ai_feedback(training_prompts)
```

**RLAIF vs RLHF 비교**:

```python
"""
RLHF vs RLAIF 비교:

┌─────────────┬─────────────┬─────────────┐
│   Aspect    │    RLHF     │   RLAIF     │
├─────────────┼─────────────┼─────────────┤
│ Feedback    │ Human       │ AI          │
│ 비용        │ $$$         │ $           │
│ 속도        │ 느림        │ 빠름        │
│ 확장성      │ 제한적      │ 무한        │
│ 일관성      │ 낮음        │ 높음        │
│ 품질        │ 최고        │ 거의 동등  │
└─────────────┴─────────────┴─────────────┘

연구 결과 (Google, 2023):
- RLAIF ≈ RLHF (성능)
- RLAIF >> RLHF (효율성)

언제 RLAIF?
- 대규모 데이터 필요
- 빠른 iteration
- 일관된 기준 필요

언제 RLHF?
- 최고 품질 요구
- 도메인 전문 지식 필요
- 미묘한 인간 선호도
"""
```

---

### 6. **Multi-Agent Systems 2.0** ⭐⭐⭐⭐⭐

**진화**: 단순 협업 → 복잡한 조직 구조

```
2023: 기본 Multi-Agent
- AutoGPT: 단일 agent가 여러 도구 사용
- BabyAGI: 작업 분해 및 실행

2024-2025: Advanced Multi-Agent
- MetaGPT: 소프트웨어 회사 시뮬레이션
- AgentVerse: 사회 시뮬레이션
- ChatDev: 전체 개발 팀

핵심 발전:
1. 역할 분화 (PM, Designer, Engineer, QA)
2. 계층 구조 (Manager-Worker)
3. 의사소통 프로토콜
4. 메모리 공유
```

**MetaGPT - Software Company Simulation**:

```python
class MetaGPT:
    """
    MetaGPT: Multi-Agent Meta Programming

    논문: "MetaGPT: Meta Programming for Multi-Agent Collaborative Framework" (2023)

    핵심: 소프트웨어 개발 프로세스를 agent로 시뮬레이션

    Roles:
    - Product Manager
    - Architect
    - Engineer
    - QA Engineer

    Process:
    User Requirement
      → PM (PRD 작성)
      → Architect (설계)
      → Engineer (구현)
      → QA (테스트)
      → Product!
    """

    def __init__(self):
        self.pm = ProductManager()
        self.architect = Architect()
        self.engineer = Engineer()
        self.qa = QAEngineer()

        self.shared_workspace = Workspace()

    def develop_software(self, user_requirement):
        """
        소프트웨어 개발 전체 프로세스

        의도: Human-like 개발 프로세스
        """

        # Phase 1: Requirements Analysis
        # PM이 PRD (Product Requirement Document) 작성
        prd = self.pm.write_prd(user_requirement)
        self.shared_workspace.add_document("PRD", prd)

        # Phase 2: System Design
        # Architect가 설계 문서 작성
        design = self.architect.design_system(prd)
        self.shared_workspace.add_document("Design", design)

        # Phase 3: Implementation
        # Engineer가 코드 작성
        code = self.engineer.implement(design)
        self.shared_workspace.add_code(code)

        # Phase 4: Testing
        # QA가 테스트 및 버그 리포트
        test_results = self.qa.test(code, prd)

        # Phase 5: Bug Fix (if needed)
        if test_results['bugs']:
            # Engineer가 버그 수정
            fixed_code = self.engineer.fix_bugs(test_results['bugs'])
            self.shared_workspace.update_code(fixed_code)

        return self.shared_workspace.get_product()


class ProductManager:
    """Product Manager Agent"""

    def write_prd(self, user_requirement):
        """
        PRD 작성

        PRD 포함 사항:
        - 기능 목록
        - 사용자 스토리
        - 성공 지표
        - 우선순위
        """
        prompt = f"""
You are a Product Manager. Write a detailed PRD for:

User Requirement:
{user_requirement}

PRD should include:
1. Goals and Objectives
2. User Stories
3. Functional Requirements
4. Non-Functional Requirements
5. Success Metrics

PRD:"""

        prd = llm.generate(prompt)
        return prd


class Architect:
    """System Architect Agent"""

    def design_system(self, prd):
        """
        시스템 설계

        포함 사항:
        - 아키텍처 다이어그램
        - 데이터 모델
        - API 설계
        - 기술 스택
        """
        prompt = f"""
You are a System Architect. Design a system based on:

PRD:
{prd}

Provide:
1. System Architecture (components, their responsibilities)
2. Data Models
3. API Endpoints
4. Technology Stack

Design Document:"""

        design = llm.generate(prompt)
        return design


class Engineer:
    """Software Engineer Agent"""

    def implement(self, design):
        """
        코드 구현

        의도: 설계를 실제 코드로
        """
        prompt = f"""
You are a Software Engineer. Implement the system based on:

Design Document:
{design}

Provide clean, well-commented code for all components.

Code:"""

        code = llm.generate(prompt)
        return code

    def fix_bugs(self, bugs):
        """버그 수정"""
        prompt = f"""
Fix the following bugs:

{bugs}

Fixed code:"""

        fixed = llm.generate(prompt)
        return fixed


class QAEngineer:
    """QA Engineer Agent"""

    def test(self, code, prd):
        """
        테스트

        의도:
        - 기능 테스트
        - 엣지 케이스
        - PRD 요구사항 충족 확인
        """
        prompt = f"""
You are a QA Engineer. Test the following code against the PRD:

Code:
{code}

PRD:
{prd}

Provide:
1. Test Cases
2. Test Results
3. Bug Reports (if any)

Test Report:"""

        test_results = llm.generate(prompt)
        return self.parse_test_results(test_results)


"""
MetaGPT의 장점:

1. 구조화된 출력
   - 각 agent가 명확한 포맷 산출
   - PRD → Design → Code → Test

2. 역할 분화
   - 각 agent가 전문 영역
   - 더 나은 품질

3. Iterative improvement
   - QA → Bug fix → Retest
   - Human-like process

4. 확장 가능
   - 새로운 role 추가 가능
   - Designer, DevOps, etc.

실제 사용:
- 프로토타입 개발
- 코드 리뷰 자동화
- 문서 생성

한계:
- 복잡한 프로젝트는 여전히 어려움
- Agent 간 miscommunication
- 비용 (여러 LLM 호출)
"""
```

**AgentVerse - Multi-Agent Collaboration**:

```python
class AgentVerse:
    """
    AgentVerse: Multi-agent collaboration framework

    논문: "AgentVerse: Facilitating Multi-Agent Collaboration" (2023)

    특징:
    - 동적 팀 구성
    - 전문가 모집 (Expert recruitment)
    - 의사결정 메커니즘
    """

    def __init__(self, agent_pool):
        """
        Args:
            agent_pool: 사용 가능한 agent들
        """
        self.agent_pool = agent_pool
        self.active_agents = []

    def solve_task(self, task):
        """
        협업으로 작업 해결

        Process:
        1. 작업 분석
        2. 필요한 전문가 모집
        3. 협업 수행
        4. 결과 통합
        """

        # Step 1: 작업 분석
        task_analysis = self.analyze_task(task)

        # Step 2: 전문가 모집
        # 의도: 작업에 맞는 agent 선택
        required_experts = task_analysis['required_skills']
        self.recruit_experts(required_experts)

        # Step 3: 작업 분해
        subtasks = self.decompose_task(task, self.active_agents)

        # Step 4: 병렬 실행
        results = {}
        for agent, subtask in zip(self.active_agents, subtasks):
            result = agent.execute(subtask)
            results[agent.role] = result

        # Step 5: 결과 통합
        final_result = self.integrate_results(results)

        return final_result

    def recruit_experts(self, required_skills):
        """
        전문가 모집

        의도: 작업에 최적화된 팀 구성
        """
        recruited = []

        for skill in required_skills:
            # Agent pool에서 해당 skill 보유 agent 찾기
            candidates = [
                agent for agent in self.agent_pool
                if skill in agent.skills
            ]

            if candidates:
                # 가장 적합한 agent 선택
                best_agent = max(candidates, key=lambda a: a.expertise[skill])
                recruited.append(best_agent)

        self.active_agents = recruited

    def horizontal_communication(self):
        """
        수평적 의사소통

        의도: Agent 간 정보 공유 및 토론
        """
        # Round-table discussion
        for round in range(3):
            for agent in self.active_agents:
                # 다른 agent의 의견 수집
                others_opinions = [
                    other.get_opinion()
                    for other in self.active_agents
                    if other != agent
                ]

                # 자신의 의견 업데이트
                agent.update_opinion(others_opinions)

        # Consensus 도출
        consensus = self.reach_consensus()
        return consensus

    def reach_consensus(self):
        """
        합의 도출

        방법:
        - Voting (투표)
        - Weighted voting (전문성 가중)
        - Debate until agreement
        """
        opinions = [agent.final_opinion for agent in self.active_agents]

        # Voting
        votes = {}
        for opinion in opinions:
            votes[opinion] = votes.get(opinion, 0) + 1

        # Majority opinion
        consensus = max(votes, key=votes.get)

        return consensus


"""
Multi-Agent의 핵심 패턴:

1. 계층적 (Hierarchical)
   Manager → Workers
   - Manager: 작업 분배, 조율
   - Workers: 실행

2. 수평적 (Horizontal)
   Peer-to-peer collaboration
   - 동등한 전문가들의 토론
   - Consensus 기반 결정

3. Workflow (순차적)
   Agent1 → Agent2 → Agent3
   - 파이프라인 (예: MetaGPT)
   - 각 단계의 출력이 다음 입력

4. Market-based
   - Task auction
   - 가장 적합한 agent가 수주

응용:
- 복잡한 문제 해결
- 창의적 brainstorming
- 시뮬레이션 (사회, 경제)
"""
```

---

이어서 나머지 트렌드들을 작성하겠습니다. 파일이 너무 길어지니 여기서 일단 저장하고 commit하겠습니다.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create current AI trends directory structure", "status": "completed", "activeForm": "Creating trends directory structure"}, {"content": "Write 2024-2025 AI landscape overview", "status": "completed", "activeForm": "Writing AI landscape overview"}, {"content": "Create research paper fundamentals guide", "status": "completed", "activeForm": "Creating research fundamentals"}, {"content": "Create 2024-2025 cutting-edge research trends", "status": "in_progress", "activeForm": "Creating research trends guide"}, {"content": "Complete research trends document (add remaining 4 trends)", "status": "pending", "activeForm": "Completing research trends"}, {"content": "Create mathematical foundations for papers", "status": "pending", "activeForm": "Creating math foundations"}, {"content": "Create Multimodal AI guide (GPT-4V, Gemini, Claude 3)", "status": "pending", "activeForm": "Creating Multimodal AI guide"}, {"content": "Create AI Agents guide (AutoGPT, LangChain)", "status": "pending", "activeForm": "Creating AI Agents guide"}, {"content": "Create LLMOps & Production guide", "status": "pending", "activeForm": "Creating LLMOps guide"}, {"content": "Create Prompt Engineering advanced guide", "status": "pending", "activeForm": "Creating Prompt Engineering guide"}, {"content": "Update README with trends section", "status": "pending", "activeForm": "Updating README"}, {"content": "Commit and push all trends documentation", "status": "pending", "activeForm": "Committing trends documentation"}]