# Advanced Prompt Engineering (고급 프롬프트 엔지니어링)

## 🎯 목표

**프롬프트 엔지니어링을 마스터하여 LLM 성능 극대화**

프롬프트는 새로운 프로그래밍 언어입니다. 잘 작성된 프롬프트는:
- 정확도 50%+ 향상
- 비용 30%+ 절감 (짧은 프롬프트)
- 일관된 출력 형식

---

## 📚 기본 원칙

### 1. Clear and Specific Instructions

```python
"""
나쁜 예:
"이 텍스트를 요약해"

좋은 예:
"다음 기술 문서를 3개 bullet point로 요약해주세요.
각 bullet point는 한 문장으로, 핵심 기술적 내용에 집중하세요."
"""

# 예제
bad_prompt = "Summarize this"

good_prompt = """
Summarize the following text in exactly 3 bullet points.
Each bullet point should:
- Be one sentence
- Start with an action verb
- Focus on key technical details

Text:
{text}

Summary:
"""
```

### 2. Provide Context and Role

```python
"""
역할 부여 (Role Prompting)

의도: LLM이 특정 전문가 관점에서 답변
"""

system_prompt = """
You are a senior software engineer with 10 years of experience in:
- Python backend development
- System design
- Database optimization

Answer questions with:
- Practical code examples
- Performance considerations
- Best practices and anti-patterns
"""

# 사용
response = llm.chat([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "How do I optimize this SQL query?"}
])
```

### 3. Show Examples (Few-Shot Learning)

```python
"""
Few-Shot Prompting

의도: 예제로 출력 형식 지정
"""

few_shot_prompt = """
Extract the following information from customer reviews:
- Sentiment: positive/negative/neutral
- Product: product name
- Issue: main complaint (if any)

Example 1:
Review: "The laptop is great but battery life is terrible"
Output:
{
  "sentiment": "negative",
  "product": "laptop",
  "issue": "battery life"
}

Example 2:
Review: "Love this phone! Camera is amazing"
Output:
{
  "sentiment": "positive",
  "product": "phone",
  "issue": null
}

Now extract from this review:
Review: "{user_review}"
Output:
"""

# 장점: 일관된 JSON 형식
```

---

## 🚀 고급 기법

### 1. Chain-of-Thought (CoT)

**개념**: "단계별로 생각하게" 만들기

```python
class ChainOfThought:
    """
    Chain-of-Thought Prompting

    논문: "Chain-of-Thought Prompting Elicits Reasoning in
           Large Language Models" (Wei et al., 2022)

    핵심: "Let's think step by step" 추가만으로 성능 향상!
    """

    @staticmethod
    def basic_cot(question):
        """
        기본 CoT

        의도: 중간 추론 과정 생성
        """
        prompt = f"""
Question: {question}

Let's approach this step-by-step:
1) First, let me identify what we're looking for
2) Then, I'll break down the problem
3) Finally, I'll calculate the answer

Answer:
"""
        return prompt

    @staticmethod
    def zero_shot_cot(question):
        """
        Zero-Shot CoT

        놀라운 발견: "Let's think step by step" 만 추가!

        성능 향상:
        - GSM8K (수학): 17% → 41%
        - CommonsenseQA: 69% → 79%
        """
        prompt = f"""
{question}

Let's think step by step:
"""
        return prompt

    @staticmethod
    def few_shot_cot(question):
        """
        Few-Shot CoT

        의도: 추론 과정 예제 제공
        """
        prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?

A: Let's think step by step:
   1) Roger started with 5 tennis balls
   2) He bought 2 cans, each with 3 balls
   3) 2 cans × 3 balls/can = 6 balls
   4) Total: 5 + 6 = 11 balls
   Answer: 11

Q: {question}

A: Let's think step by step:
"""
        return prompt


# 실제 사용
cot = ChainOfThought()

question = "A store has 15 apples. They sell 40% and buy 12 more. How many now?"

# Zero-shot CoT
prompt = cot.zero_shot_cot(question)
response = llm.generate(prompt)

"""
Expected output:
Let's think step by step:
1) Start with 15 apples
2) Sell 40%: 15 × 0.4 = 6 apples sold
3) Remaining: 15 - 6 = 9 apples
4) Buy 12 more: 9 + 12 = 21 apples
Answer: 21 apples
"""
```

### 2. Self-Consistency

**개념**: 여러 번 답변 → 다수결

```python
class SelfConsistency:
    """
    Self-Consistency

    논문: "Self-Consistency Improves Chain of Thought Reasoning"
           (Wang et al., 2022)

    방법:
    1. 같은 질문을 N번 (다른 추론 경로)
    2. 가장 많이 나온 답 선택

    성능 향상:
    - GSM8K: 74% → 82%
    """

    def __init__(self, llm, num_samples=5):
        self.llm = llm
        self.num_samples = num_samples

    def generate_consistent(self, question):
        """
        Self-Consistency로 답변 생성

        의도: 정확도 향상 (다수결)
        """
        prompt = f"""
{question}

Let's think step by step:
"""

        # 여러 경로 생성
        # 의도: temperature > 0으로 다양성 확보
        responses = []
        for _ in range(self.num_samples):
            response = self.llm.generate(
                prompt,
                temperature=0.7  # 다양한 추론 경로
            )
            responses.append(response)

        # 최종 답 추출
        final_answers = [
            self.extract_final_answer(r)
            for r in responses
        ]

        # 다수결
        from collections import Counter
        answer_counts = Counter(final_answers)
        most_common = answer_counts.most_common(1)[0][0]

        return {
            'final_answer': most_common,
            'confidence': answer_counts[most_common] / self.num_samples,
            'all_answers': final_answers
        }

    def extract_final_answer(self, response):
        """
        응답에서 최종 답 추출

        방법:
        - "Answer: " 이후 텍스트
        - 정규식
        - LLM으로 추출
        """
        import re

        # "Answer: X" 패턴 찾기
        match = re.search(r'Answer:\s*(.+)', response)
        if match:
            return match.group(1).strip()

        # 마지막 줄
        return response.strip().split('\n')[-1]


# 사용 예제
sc = SelfConsistency(llm, num_samples=5)

question = """
A farmer has 17 sheep. All but 9 die. How many are left?
"""

result = sc.generate_consistent(question)

print(f"Final Answer: {result['final_answer']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All answers: {result['all_answers']}")

# Output:
# Final Answer: 9
# Confidence: 100%
# All answers: [9, 9, 9, 9, 9]
```

### 3. Tree of Thoughts (ToT)

**개념**: 여러 사고 경로를 트리로 탐색

```python
class TreeOfThoughts:
    """
    Tree of Thoughts

    논문: "Tree of Thoughts: Deliberate Problem Solving with LLMs"
           (Yao et al., 2023)

    vs Chain-of-Thought:
    - CoT: 선형 (A → B → C)
    - ToT: 트리 (A → B1/B2/B3 → C)

    언제 사용?
    - 창의적 문제 (여러 접근)
    - 복잡한 계획
    - 게임 (체스, 24 game)
    """

    def __init__(self, llm):
        self.llm = llm

    def solve(self, problem, depth=3, breadth=3):
        """
        ToT로 문제 해결

        Args:
            depth: 탐색 깊이
            breadth: 각 단계 후보 수

        의도: BFS로 사고 공간 탐색
        """
        # Root node
        root = {
            'state': problem,
            'path': [],
            'value': 0
        }

        # BFS
        current_level = [root]

        for level in range(depth):
            next_level = []

            for node in current_level:
                # 각 노드에서 breadth개 후보 생성
                candidates = self.generate_thoughts(
                    node['state'],
                    k=breadth
                )

                # 각 후보 평가
                for candidate in candidates:
                    value = self.evaluate_thought(candidate)

                    next_node = {
                        'state': candidate['next_state'],
                        'path': node['path'] + [candidate['thought']],
                        'value': value
                    }
                    next_level.append(next_node)

            # Top-k 선택 (pruning)
            # 의도: 유망한 경로만 탐색
            next_level.sort(key=lambda x: x['value'], reverse=True)
            current_level = next_level[:breadth]

        # 최선 경로 반환
        best = max(current_level, key=lambda x: x['value'])
        return best

    def generate_thoughts(self, state, k=3):
        """
        k개의 다음 사고 생성

        의도: 다양한 접근 시도
        """
        prompt = f"""
Current problem state:
{state}

Generate {k} different next steps to solve this problem.
Each step should explore a different approach.

Next steps:
"""

        response = self.llm.generate(prompt, temperature=0.8)

        # Parse k개 thoughts
        thoughts = self.parse_thoughts(response, k)

        return thoughts

    def evaluate_thought(self, thought):
        """
        사고 평가

        방법:
        1. LLM self-evaluation
        2. Heuristic
        3. Simulation (게임의 경우)
        """
        prompt = f"""
Evaluate the following reasoning step on a scale of 1-10:

{thought}

Consider:
- Correctness
- Promising direction
- Novelty

Score (1-10):
"""

        score = self.llm.generate(prompt, max_tokens=2)
        return float(score)


# 실제 예제: 24 Game
tot = TreeOfThoughts(llm)

problem = """
Use numbers [4, 9, 10, 13] to make 24.
You can use +, -, *, / operators.
Each number must be used exactly once.
"""

solution = tot.solve(problem, depth=3, breadth=3)

print("Solution path:")
for i, step in enumerate(solution['path']):
    print(f"  Step {i+1}: {step}")
print(f"\nFinal answer: {solution['state']}")
print(f"Confidence: {solution['value']}/10")

# Expected output:
# Step 1: Try (13 - 9) first
# Step 2: Then (10 - 4)
# Step 3: Multiply: (13-9) * (10-4) = 4 * 6 = 24
```

### 4. ReAct (Reasoning + Acting)

**개념**: 추론과 행동을 번갈아 수행

```python
class ReAct:
    """
    ReAct: Reasoning and Acting

    논문: "ReAct: Synergizing Reasoning and Acting in Language Models"
           (Yao et al., 2022)

    패턴:
    Thought → Action → Observation → Thought → ...

    응용:
    - 웹 검색 + QA
    - 코드 실행 + 디버깅
    - API 호출 + 처리
    """

    def __init__(self, llm, tools):
        """
        Args:
            llm: Language model
            tools: 사용 가능한 도구들
        """
        self.llm = llm
        self.tools = tools

    def solve(self, question, max_steps=5):
        """
        ReAct loop

        의도: 생각 → 행동 → 관찰 반복
        """
        trajectory = []

        for step in range(max_steps):
            # Thought: 다음에 무엇을 할지 추론
            thought = self.generate_thought(question, trajectory)
            trajectory.append(('Thought', thought))

            # Action: 도구 선택 및 실행
            action = self.generate_action(thought)

            if action['type'] == 'Finish':
                # 최종 답변
                return {
                    'answer': action['value'],
                    'trajectory': trajectory
                }

            trajectory.append(('Action', action))

            # Observation: 행동 결과
            observation = self.execute_action(action)
            trajectory.append(('Observation', observation))

        # Max steps 도달
        return {
            'answer': "Could not solve within max steps",
            'trajectory': trajectory
        }

    def generate_thought(self, question, trajectory):
        """
        현재 상황 분석 및 다음 행동 계획
        """
        # Trajectory를 텍스트로 변환
        history = self.format_trajectory(trajectory)

        prompt = f"""
Question: {question}

{history}

Thought: Let me think about what to do next.
"""

        thought = self.llm.generate(prompt, max_tokens=100)
        return thought

    def generate_action(self, thought):
        """
        Thought → Action

        Action types:
        - Search: 웹 검색
        - Lookup: 문서에서 찾기
        - Calculate: 계산
        - Finish: 최종 답변
        """
        prompt = f"""
{thought}

What action should I take?

Available actions:
- Search[query]: Search the web
- Lookup[term]: Find in current document
- Calculate[expression]: Do math
- Finish[answer]: Give final answer

Action:
"""

        action_str = self.llm.generate(prompt, max_tokens=50)

        # Parse action
        action = self.parse_action(action_str)
        return action

    def execute_action(self, action):
        """
        Action 실행

        의도: 실제 도구 호출
        """
        tool_name = action['type']
        tool_input = action['value']

        if tool_name in self.tools:
            result = self.tools[tool_name](tool_input)
            return result
        else:
            return f"Tool {tool_name} not found"


# 도구 정의
tools = {
    'Search': lambda query: search_web(query),
    'Lookup': lambda term: lookup_in_doc(term),
    'Calculate': lambda expr: eval(expr),
    'Finish': lambda answer: answer
}

# 사용 예제
react = ReAct(llm, tools)

question = """
What is the elevation of the highest point in the country
where the 2024 Olympics were held?
"""

result = react.solve(question)

print("Trajectory:")
for step_type, content in result['trajectory']:
    print(f"{step_type}: {content}\n")

print(f"Final Answer: {result['answer']}")

"""
Expected trajectory:

Thought: I need to first find where the 2024 Olympics were held.

Action: Search["2024 Olympics location"]

Observation: The 2024 Summer Olympics were held in Paris, France.

Thought: Now I need to find the highest point in France.

Action: Search["highest point in France elevation"]

Observation: Mont Blanc is the highest point in France at 4,808 meters.

Action: Finish["4,808 meters"]

Final Answer: 4,808 meters
"""
```

---

## 🎨 실전 패턴

### 1. Structured Output (JSON)

```python
class StructuredOutput:
    """
    일관된 JSON 출력 보장

    방법:
    1. JSON schema 제공
    2. Few-shot examples
    3. Constrained decoding (일부 모델)
    """

    @staticmethod
    def extract_entities(text):
        """
        엔티티 추출 → JSON
        """
        prompt = f"""
Extract named entities from the text and return as JSON.

JSON Schema:
{{
  "persons": [string],
  "organizations": [string],
  "locations": [string],
  "dates": [string]
}}

Example:
Text: "Apple CEO Tim Cook visited Paris on June 5th"
Output:
{{
  "persons": ["Tim Cook"],
  "organizations": ["Apple"],
  "locations": ["Paris"],
  "dates": ["June 5th"]
}}

Now extract from:
Text: "{text}"
Output (JSON only, no explanation):
"""

        response = llm.generate(prompt, temperature=0)

        # Parse JSON
        import json
        try:
            entities = json.loads(response)
            return entities
        except:
            # Retry or error handling
            return None

    @staticmethod
    def function_calling_pattern(user_query):
        """
        Function Calling Pattern (OpenAI)

        의도: LLM이 함수 호출 결정
        """
        # 사용 가능한 함수 정의
        functions = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "get_stock_price",
                "description": "Get current stock price",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        ]

        # OpenAI function calling
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "user", "content": user_query}
            ],
            functions=functions,
            function_call="auto"
        )

        # LLM이 함수 호출 결정
        message = response.choices[0].message

        if message.function_call:
            # 함수 실행
            function_name = message.function_call.name
            arguments = json.loads(message.function_call.arguments)

            # 실제 함수 호출
            if function_name == "get_weather":
                result = get_weather(**arguments)
            elif function_name == "get_stock_price":
                result = get_stock_price(**arguments)

            return result
        else:
            # 일반 답변
            return message.content


# 사용
query = "What's the weather in Seoul?"
result = StructuredOutput.function_calling_pattern(query)
# LLM이 자동으로 get_weather("Seoul") 호출
```

### 2. Prompt Chaining

```python
class PromptChaining:
    """
    Prompt Chaining

    의도: 복잡한 작업을 단계로 분해

    장점:
    - 각 단계 최적화 가능
    - 중간 결과 검증
    - 비용 절감 (필요한 단계만)
    """

    def __init__(self, llm):
        self.llm = llm

    def write_blog_post(self, topic):
        """
        블로그 작성 파이프라인

        Steps:
        1. 아웃라인 생성
        2. 각 섹션 작성
        3. 서론/결론 추가
        4. 편집 및 다듬기
        """

        # Step 1: Outline
        outline = self.generate_outline(topic)

        # Step 2: Write sections
        sections = []
        for section_title in outline['sections']:
            content = self.write_section(
                topic,
                section_title,
                outline['key_points']
            )
            sections.append(content)

        # Step 3: Introduction
        introduction = self.write_introduction(topic, outline)

        # Step 4: Conclusion
        conclusion = self.write_conclusion(topic, sections)

        # Step 5: Final editing
        full_post = '\n\n'.join([introduction] + sections + [conclusion])
        edited = self.edit_post(full_post)

        return edited

    def generate_outline(self, topic):
        """Step 1: 아웃라인"""
        prompt = f"""
Create a detailed outline for a blog post about: {topic}

Include:
- 3-5 main sections
- Key points for each section
- Target word count: 1000-1500 words

Outline:
"""
        response = self.llm.generate(prompt)
        return self.parse_outline(response)

    def write_section(self, topic, section_title, key_points):
        """Step 2: 섹션 작성"""
        prompt = f"""
Write a blog post section about: {section_title}

Context: This is part of a post about {topic}

Key points to cover:
{chr(10).join(f'- {p}' for p in key_points)}

Write 200-300 words in an engaging, informative style.

Section:
"""
        return self.llm.generate(prompt)

    def edit_post(self, post):
        """Step 5: 편집"""
        prompt = f"""
Edit the following blog post for:
- Grammar and spelling
- Clarity and flow
- Engaging language
- SEO optimization

Original:
{post}

Edited version:
"""
        return self.llm.generate(prompt)


# 사용
chaining = PromptChaining(llm)
blog_post = chaining.write_blog_post("The Future of AI in Healthcare")
```

### 3. Prompt Compression

```python
class PromptCompression:
    """
    Prompt 압축

    목적: 토큰 수 줄여 비용 절감

    방법:
    1. 불필요한 단어 제거
    2. 약어 사용
    3. Few-shot → Zero-shot 전환
    4. LLMLingua (자동 압축)
    """

    @staticmethod
    def compress_manual(prompt):
        """
        수동 압축

        Before: 100 tokens
        After: 60 tokens (-40%)
        """
        # Before
        verbose = """
        Please carefully analyze the following customer review
        and extract the sentiment (positive, negative, or neutral),
        the product that is being reviewed, and any specific issues
        or complaints that the customer mentioned.
        """

        # After
        compressed = """
        Extract from review:
        - Sentiment: positive/negative/neutral
        - Product: name
        - Issues: list
        """

        # 40% 토큰 절감!
        return compressed

    @staticmethod
    def compress_with_llm(long_prompt):
        """
        LLM으로 압축

        의도: 의미 보존하며 압축
        """
        compression_prompt = f"""
Compress the following prompt to use fewer tokens while preserving meaning:

Original prompt:
{long_prompt}

Compressed prompt (use abbreviations, remove filler words):
"""

        compressed = llm.generate(compression_prompt)
        return compressed

    @staticmethod
    def llmlingua_compression(prompt):
        """
        LLMLingua: 자동 압축

        논문: "LLMLingua: Compressing Prompts for Accelerated
               Inference of Large Language Models"

        방법:
        - 중요하지 않은 토큰 제거
        - Perplexity 기반
        - 2-5배 압축 가능
        """
        from llmlingua import PromptCompressor

        compressor = PromptCompressor()

        compressed_prompt = compressor.compress_prompt(
            prompt,
            target_token=100,  # 목표 토큰 수
            rate=0.5  # 압축률
        )

        return compressed_prompt['compressed_prompt']


# 비용 비교
original_tokens = 500
compressed_tokens = 200

cost_original = original_tokens / 1000 * 0.01  # GPT-4
cost_compressed = compressed_tokens / 1000 * 0.01

savings = (cost_original - cost_compressed) / cost_original

print(f"Token reduction: {(1 - compressed_tokens/original_tokens)*100:.0f}%")
print(f"Cost reduction: {savings*100:.0f}%")
# Output: 60% token reduction, 60% cost reduction
```

---

## 🔧 실전 도구

### Prompt Management

```python
class PromptManager:
    """
    프롬프트 버전 관리

    의도: 프롬프트를 코드처럼 관리
    """

    def __init__(self):
        self.prompts = {}
        self.versions = {}

    def register(self, name, template, version="v1"):
        """프롬프트 등록"""
        if name not in self.prompts:
            self.prompts[name] = {}
            self.versions[name] = []

        self.prompts[name][version] = template
        self.versions[name].append(version)

    def get(self, name, version="latest", **kwargs):
        """프롬프트 가져오기"""
        if version == "latest":
            version = self.versions[name][-1]

        template = self.prompts[name][version]
        return template.format(**kwargs)

    def a_b_test(self, name, versions, test_cases):
        """A/B 테스트"""
        results = {}

        for version in versions:
            scores = []
            for test_case in test_cases:
                prompt = self.get(name, version, **test_case['input'])
                response = llm.generate(prompt)
                score = evaluate(response, test_case['expected'])
                scores.append(score)

            results[version] = {
                'avg_score': np.mean(scores),
                'scores': scores
            }

        return results


# 사용 예제
pm = PromptManager()

# 버전 1
pm.register(
    "sentiment_analysis",
    """
Analyze sentiment of: "{text}"
Return: positive/negative/neutral
""",
    version="v1"
)

# 버전 2 (개선)
pm.register(
    "sentiment_analysis",
    """
Sentiment analysis of customer review:

Review: "{text}"

Classification:
- positive (satisfied, happy)
- negative (unsatisfied, complaint)
- neutral (factual, no emotion)

Also provide confidence score (0-1).

Output (JSON):
{{"sentiment": "...", "confidence": 0.0}}
""",
    version="v2"
)

# A/B 테스트
test_cases = [
    {
        'input': {'text': 'Great product!'},
        'expected': 'positive'
    },
    # ...
]

results = pm.a_b_test(
    "sentiment_analysis",
    versions=["v1", "v2"],
    test_cases=test_cases
)

print(f"v1 accuracy: {results['v1']['avg_score']:.2%}")
print(f"v2 accuracy: {results['v2']['avg_score']:.2%}")
```

---

## 📊 Evaluation & Testing

```python
class PromptEvaluator:
    """
    프롬프트 평가

    지표:
    1. Accuracy (정확도)
    2. Consistency (일관성)
    3. Latency (응답 시간)
    4. Cost (비용)
    """

    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, prompt_template, test_set):
        """
        종합 평가
        """
        results = {
            'accuracy': [],
            'latency': [],
            'cost': [],
            'consistency': []
        }

        for test_case in test_set:
            # Prompt 생성
            prompt = prompt_template.format(**test_case['input'])

            # 생성
            import time
            start = time.time()
            response = self.llm.generate(prompt)
            latency = time.time() - start

            # 평가
            is_correct = self.check_correctness(
                response,
                test_case['expected']
            )

            # 비용 계산
            tokens = len(prompt.split()) + len(response.split())
            cost = tokens / 1000 * 0.01  # GPT-4 가격

            results['accuracy'].append(1 if is_correct else 0)
            results['latency'].append(latency)
            results['cost'].append(cost)

        # 일관성: 같은 입력을 5번
        consistency_score = self.test_consistency(
            prompt_template,
            test_set[0]  # 첫 테스트 케이스
        )

        return {
            'accuracy': np.mean(results['accuracy']),
            'avg_latency': np.mean(results['latency']),
            'total_cost': sum(results['cost']),
            'consistency': consistency_score
        }

    def test_consistency(self, prompt_template, test_case, n=5):
        """
        일관성 테스트

        의도: 같은 입력 → 같은 출력?
        """
        prompt = prompt_template.format(**test_case['input'])

        responses = []
        for _ in range(n):
            response = self.llm.generate(prompt, temperature=0)
            responses.append(response)

        # 모두 동일한지 확인
        unique_responses = len(set(responses))
        consistency = 1 - (unique_responses - 1) / n

        return consistency


# 사용
evaluator = PromptEvaluator(llm)

test_set = [
    {
        'input': {'text': 'I love this!'},
        'expected': 'positive'
    },
    {
        'input': {'text': 'Terrible experience'},
        'expected': 'negative'
    },
    # ... more tests
]

metrics = evaluator.evaluate(prompt_template, test_set)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Avg Latency: {metrics['avg_latency']:.2f}s")
print(f"Total Cost: ${metrics['total_cost']:.4f}")
print(f"Consistency: {metrics['consistency']:.2%}")
```

---

## 💡 Best Practices

### DO's ✅

```python
"""
1. 명확하고 구체적으로

Good:
"Summarize this article in 3 bullet points,
 each focusing on one key finding."

Bad:
"Summarize this."


2. 예제 제공 (Few-Shot)

Good:
Input: "Great!" → Output: "positive"
Input: "Bad" → Output: "negative"
Now classify: {text}

Bad:
Classify sentiment: {text}


3. 출력 형식 지정

Good:
"Return as JSON: {"sentiment": "...", "score": 0.0}"

Bad:
"Tell me the sentiment and score"


4. Role/Persona 부여

Good:
"You are an expert Python developer..."

Bad:
(no context)


5. 단계별 지시 (CoT)

Good:
"Let's solve this step by step:
 1) First, identify...
 2) Then, calculate...
 3) Finally, verify..."

Bad:
"Solve this math problem"
"""
```

### DON'Ts ❌

```python
"""
1. 모호한 지시 피하기

Bad:
"Make it better"
"Improve this"

Good:
"Improve clarity by:
 - Using shorter sentences
 - Adding specific examples
 - Removing jargon"


2. 너무 긴 프롬프트 피하기

Bad:
3000 token 프롬프트 (비용 ↑, 집중력 ↓)

Good:
- 핵심만 간결하게
- 또는 Prompt Chaining으로 분할


3. 일관성 없는 형식

Bad:
때로는 JSON, 때로는 plain text

Good:
항상 동일한 형식 요구


4. Leading questions

Bad:
"This is clearly positive, right?"
(bias 유발)

Good:
"What is the sentiment?"
(중립적)


5. 테스트 없이 프로덕션 배포

Bad:
작성 후 바로 배포

Good:
- Test set으로 평가
- A/B 테스트
- Gradual rollout
"""
```

---

## 📚 Resources

### Tools
```
1. PromptLayer - Prompt versioning & tracking
2. LangSmith - LangChain debugging
3. Helicone - Monitoring & analytics
4. Weights & Biases Prompts - Experiment tracking
```

### Prompt Libraries
```
1. Awesome ChatGPT Prompts (GitHub)
2. PromptBase - Marketplace
3. ShareGPT - Community prompts
4. LangChain Hub - Curated prompts
```

### Papers
```
1. Chain-of-Thought Prompting (Wei et al., 2022)
2. ReAct (Yao et al., 2022)
3. Tree of Thoughts (Yao et al., 2023)
4. Self-Consistency (Wang et al., 2022)
```

---

## ⏭️ Next Steps

프롬프트 엔지니어링을 마스터했다면:

1. **LLMOps**: 프로덕션 배포 및 모니터링
2. **Fine-tuning**: Prompt로 부족할 때
3. **AI Agents**: 도구를 사용하는 자율 시스템

👉 Continue to **04-llmops-production.md**

**Advanced Prompt Engineering 완료!** 🎉
