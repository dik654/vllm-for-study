# AI Agents 실전 가이드 (2024-2025)

## 목차
1. [개요](#개요)
2. [LangChain Agents](#langchain-agents)
3. [AutoGPT & Autonomous Agents](#autogpt--autonomous-agents)
4. [CrewAI - Multi-Agent Collaboration](#crewai---multi-agent-collaboration)
5. [고급 패턴](#고급-패턴)
6. [프로덕션 배포](#프로덕션-배포)

---

## 개요

### AI Agent란?
**정의**: 목표를 받아 스스로 계획하고, 도구를 사용하며, 피드백을 반영하여 작업을 완수하는 AI 시스템

**전통적 LLM vs Agent**
```
전통적 LLM:
User: "파리의 날씨를 알려줘"
LLM: "죄송합니다. 실시간 정보에 접근할 수 없습니다."

Agent:
User: "파리의 날씨를 알려줘"
Agent: [Thought] 날씨 API를 호출해야겠다
       [Action] weather_api.get("Paris")
       [Observation] 현재 15°C, 맑음
       [Answer] "파리는 현재 15°C이며 맑습니다."
```

### Agent 핵심 구성 요소

1. **Planning (계획)**: 목표 → 하위 작업 분해
2. **Tool Use (도구 사용)**: API, 검색, 코드 실행
3. **Memory (기억)**: 과거 대화/행동 기억
4. **Reflection (성찰)**: 실수 인지 및 개선

### 2024-2025 주요 프레임워크

| 프레임워크 | 특징 | 사용 사례 |
|-----------|------|----------|
| **LangChain** | 범용성, 풍부한 통합 | RAG, 데이터 분석, 고객 지원 |
| **AutoGPT** | 자율성, 장기 목표 수행 | 리서치, 콘텐츠 생성, 프로젝트 관리 |
| **CrewAI** | 멀티 에이전트 협업 | 소프트웨어 개발, 마케팅 캠페인 |
| **LangGraph** | 그래프 기반 워크플로우 | 복잡한 워크플로우, 조건부 실행 |
| **AutoGen** | Microsoft의 멀티 에이전트 | 코드 생성, 디버깅, 협업 |

---

## LangChain Agents

### 1. 기본 ReAct Agent

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub

class BasicAgent:
    """
    ReAct 패턴: Reasoning + Acting

    루프:
    1. Thought: 다음에 무엇을 할지 생각
    2. Action: 도구 선택 및 실행
    3. Observation: 결과 관찰
    4. 반복 or 답변
    """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

        # 도구 정의
        self.tools = [
            Tool(
                name="Calculator",
                func=self.calculator,
                description="수학 계산. 입력: '3 * 4'"
            ),
            Tool(
                name="Search",
                func=self.search,
                description="웹 검색. 입력: 검색 쿼리"
            ),
            Tool(
                name="WeatherAPI",
                func=self.get_weather,
                description="날씨 조회. 입력: 도시 이름"
            )
        ]

        # ReAct 프롬프트 (LangChain hub에서)
        prompt = hub.pull("hwchase17/react")

        # Agent 생성
        agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )

    def calculator(self, expression: str) -> str:
        """계산기 도구"""
        try:
            result = eval(expression)
            return f"결과: {result}"
        except Exception as e:
            return f"오류: {str(e)}"

    def search(self, query: str) -> str:
        """검색 도구 (Tavily API 사용)"""
        from langchain_community.tools.tavily_search import TavilySearchResults

        search_tool = TavilySearchResults(max_results=3)
        results = search_tool.invoke(query)
        return str(results)

    def get_weather(self, city: str) -> str:
        """날씨 API (예시)"""
        import requests

        # OpenWeatherMap API 예시
        api_key = "YOUR_API_KEY"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            return f"{city}의 현재 날씨: {temp}°C, {desc}"
        else:
            return "날씨 정보를 가져올 수 없습니다."

    def run(self, task: str):
        """Agent 실행"""
        result = self.agent_executor.invoke({"input": task})
        return result['output']

# 실전 사용
agent = BasicAgent()

# 예시 1: 복합 작업
result = agent.run("파리의 날씨를 알려주고, 화씨로 변환해줘")
# Thought: 먼저 파리 날씨를 확인해야겠다
# Action: WeatherAPI("Paris")
# Observation: 파리의 현재 날씨: 15°C, 맑음
# Thought: 이제 섭씨를 화씨로 변환해야겠다
# Action: Calculator("15 * 9/5 + 32")
# Observation: 결과: 59.0
# Answer: 파리는 현재 15°C (59°F)이며 맑습니다.

# 예시 2: 리서치 작업
result = agent.run("2024년 노벨 물리학상 수상자를 찾고, 그들의 업적을 요약해줘")
# Thought: 검색으로 최신 정보를 찾아야겠다
# Action: Search("2024 노벨 물리학상 수상자")
# Observation: [검색 결과...]
# Answer: 2024년 노벨 물리학상은...
```

### 2. Custom Tools - 데이터베이스 Agent

```python
from langchain.agents import tool
import sqlite3
import pandas as pd

class DatabaseAgent:
    """
    데이터베이스 질의 Agent

    기능:
    - 자연어 → SQL 변환
    - 쿼리 실행
    - 결과 해석
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

        # Tools 정의
        tools = [
            self.get_schema_tool(),
            self.execute_query_tool(),
            self.get_table_info_tool()
        ]

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )

    @tool
    def get_schema_tool(self) -> str:
        """데이터베이스 스키마 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        schema = "데이터베이스 스키마:\n"
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            schema += f"\n테이블: {table_name}\n"
            for col in columns:
                schema += f"  - {col[1]} ({col[2]})\n"

        conn.close()
        return schema

    @tool
    def execute_query_tool(self, query: str) -> str:
        """SQL 쿼리 실행. 입력: SQL 쿼리"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()

            # 결과를 문자열로 변환
            if len(df) > 10:
                result = df.head(10).to_string() + f"\n... (총 {len(df)}개 행)"
            else:
                result = df.to_string()

            return result
        except Exception as e:
            return f"쿼리 오류: {str(e)}"

    @tool
    def get_table_info_tool(self, table_name: str) -> str:
        """특정 테이블 정보 조회. 입력: 테이블 이름"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn)
        conn.close()

        return f"테이블 {table_name}의 샘플 데이터:\n{df.to_string()}"

    def query(self, question: str):
        """자연어 질문 처리"""
        result = self.agent_executor.invoke({
            "input": f"""
            다음 질문에 답하기 위해 데이터베이스를 조회해주세요:

            {question}

            단계:
            1. get_schema_tool로 스키마 확인
            2. 필요한 경우 get_table_info_tool로 샘플 데이터 확인
            3. SQL 쿼리 작성 및 execute_query_tool로 실행
            4. 결과 해석 및 답변
            """
        })
        return result['output']

# 실전 사용
db_agent = DatabaseAgent("sales.db")

# 예시 1: 집계 쿼리
answer = db_agent.query("2024년 총 매출액은 얼마인가요?")
# Thought: 스키마를 먼저 확인해야겠다
# Action: get_schema_tool()
# Observation: 테이블 sales에 date, amount 컬럼이 있음
# Thought: 2024년 매출을 집계하는 SQL을 작성하자
# Action: execute_query_tool("SELECT SUM(amount) FROM sales WHERE strftime('%Y', date) = '2024'")
# Observation: 결과: 1,234,567
# Answer: 2024년 총 매출액은 $1,234,567입니다.

# 예시 2: 복잡한 분석
answer = db_agent.query("어떤 제품이 가장 많이 팔렸고, 월별 트렌드는 어떤가요?")
```

### 3. Memory - 대화 기억

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.agents import initialize_agent, AgentType

class MemoryAgent:
    """
    메모리를 가진 Agent

    메모리 타입:
    1. ConversationBufferMemory: 전체 대화 저장
    2. ConversationSummaryMemory: 요약 저장 (토큰 절약)
    3. ConversationBufferWindowMemory: 최근 K턴만 저장
    """
    def __init__(self, memory_type="buffer"):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)

        # 메모리 선택
        if memory_type == "buffer":
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        elif memory_type == "summary":
            memory = ConversationSummaryMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=True
            )
        else:
            from langchain.memory import ConversationBufferWindowMemory
            memory = ConversationBufferWindowMemory(
                k=5,  # 최근 5턴
                memory_key="chat_history",
                return_messages=True
            )

        # Tools
        tools = [
            Tool(name="Search", func=self.search, description="웹 검색"),
            Tool(name="Calculator", func=lambda x: str(eval(x)), description="계산")
        ]

        # Agent 생성
        self.agent = initialize_agent(
            tools,
            self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )

    def search(self, query):
        """검색 도구"""
        from langchain_community.tools.tavily_search import TavilySearchResults
        search = TavilySearchResults(max_results=2)
        return search.invoke(query)

    def chat(self, message):
        """대화"""
        return self.agent.invoke({"input": message})['output']

# 실전 사용
agent = MemoryAgent(memory_type="summary")

# 대화 1
agent.chat("내 이름은 John이야")
# "안녕하세요 John! 무엇을 도와드릴까요?"

# 대화 2 (메모리 활용)
agent.chat("내 이름이 뭐였지?")
# "당신의 이름은 John입니다."

# 대화 3 (컨텍스트 유지)
agent.chat("파이썬에서 리스트를 정렬하는 방법을 알려줘")
# [Action] Search("python list sorting")
# "파이썬에서는 .sort()나 sorted()를 사용할 수 있습니다..."

# 대화 4 (이전 대화 참조)
agent.chat("그걸 역순으로 하려면?")
# "reverse=True 파라미터를 추가하면 됩니다. 예: my_list.sort(reverse=True)"
```

---

## AutoGPT & Autonomous Agents

### 1. AutoGPT 핵심 개념

```python
import json
from typing import List, Dict

class AutoGPT:
    """
    AutoGPT: 자율적인 목표 달성 Agent

    특징:
    1. 장기 목표 수행
    2. 자가 피드백 (Self-Reflection)
    3. 계획 수정
    4. 파일 시스템 접근
    """
    def __init__(self, goal: str, max_iterations=20):
        self.goal = goal
        self.max_iterations = max_iterations
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)

        self.memory = []  # 과거 행동 기록
        self.tasks = []   # 하위 작업 목록

    def plan(self) -> List[str]:
        """
        목표 → 하위 작업 분해

        의도: 복잡한 목표를 관리 가능한 작업으로 나눔
        """
        prompt = f"""
        목표: {self.goal}

        이 목표를 달성하기 위한 구체적인 하위 작업 목록을 생성해주세요.
        각 작업은 명확하고 실행 가능해야 합니다.

        JSON 형식으로 반환:
        {{
            "tasks": [
                {{"id": 1, "description": "...", "dependencies": []}},
                {{"id": 2, "description": "...", "dependencies": [1]}}
            ]
        }}
        """

        response = self.llm.invoke(prompt)
        plan = json.loads(response.content)
        self.tasks = plan['tasks']
        return self.tasks

    def execute_task(self, task: Dict) -> Dict:
        """
        단일 작업 실행
        """
        prompt = f"""
        작업: {task['description']}

        과거 실행 내역:
        {json.dumps(self.memory[-5:], indent=2)}

        이 작업을 수행하기 위한 구체적인 액션을 선택하고 실행해주세요.

        사용 가능한 액션:
        1. search: 웹 검색
        2. write_file: 파일 작성
        3. read_file: 파일 읽기
        4. execute_code: 코드 실행
        5. task_complete: 작업 완료

        JSON 형식:
        {{
            "action": "액션 이름",
            "parameters": {{}},
            "reasoning": "왜 이 액션을 선택했는지"
        }}
        """

        response = self.llm.invoke(prompt)
        action_plan = json.loads(response.content)

        # 액션 실행
        result = self.execute_action(action_plan)

        # 메모리에 기록
        self.memory.append({
            'task': task,
            'action': action_plan,
            'result': result
        })

        return result

    def execute_action(self, action_plan: Dict):
        """
        액션 실행
        """
        action = action_plan['action']
        params = action_plan['parameters']

        if action == "search":
            from langchain_community.tools.tavily_search import TavilySearchResults
            search = TavilySearchResults(max_results=3)
            return search.invoke(params.get('query', ''))

        elif action == "write_file":
            with open(params['filename'], 'w') as f:
                f.write(params['content'])
            return f"파일 작성 완료: {params['filename']}"

        elif action == "read_file":
            with open(params['filename'], 'r') as f:
                content = f.read()
            return content

        elif action == "execute_code":
            # 주의: 실제 프로덕션에서는 샌드박스 환경 필요!
            try:
                result = exec(params['code'])
                return str(result)
            except Exception as e:
                return f"코드 실행 오류: {str(e)}"

        elif action == "task_complete":
            return "작업 완료"

        else:
            return f"알 수 없는 액션: {action}"

    def reflect(self) -> str:
        """
        Self-Reflection: 진행 상황 평가 및 계획 수정

        의도: 막혔거나 실패한 경우 다른 접근 시도
        """
        prompt = f"""
        목표: {self.goal}

        현재까지의 진행:
        {json.dumps(self.memory, indent=2)}

        질문:
        1. 목표에 얼마나 가까워졌나요? (0-100%)
        2. 현재 접근법이 올바른가요?
        3. 수정이 필요한가요?

        JSON 형식:
        {{
            "progress": 75,
            "is_on_track": true,
            "adjustments": "..."
        }}
        """

        response = self.llm.invoke(prompt)
        reflection = json.loads(response.content)
        return reflection

    def run(self):
        """
        메인 루프
        """
        # Step 1: 계획 수립
        print(f"목표: {self.goal}")
        print("계획 수립 중...")
        tasks = self.plan()
        print(f"총 {len(tasks)}개 작업 생성됨")

        # Step 2: 작업 실행
        for i in range(self.max_iterations):
            # 다음 작업 선택 (의존성 고려)
            available_tasks = [
                t for t in tasks
                if t.get('status') != 'completed'
                and all(
                    next((x for x in tasks if x['id'] == dep), {}).get('status') == 'completed'
                    for dep in t.get('dependencies', [])
                )
            ]

            if not available_tasks:
                print("모든 작업 완료!")
                break

            current_task = available_tasks[0]
            print(f"\n[반복 {i+1}] 작업 실행: {current_task['description']}")

            # 작업 실행
            result = self.execute_task(current_task)
            print(f"결과: {result}")

            # 작업 완료 표시
            current_task['status'] = 'completed'

            # Step 3: Reflection (5번마다)
            if (i + 1) % 5 == 0:
                print("\n[성찰 중...]")
                reflection = self.reflect()
                print(f"진행률: {reflection['progress']}%")

                if not reflection['is_on_track']:
                    print(f"계획 수정: {reflection['adjustments']}")
                    # 계획 재수립
                    tasks = self.plan()

        return self.memory

# 실전 사용
agent = AutoGPT(goal="파이썬으로 간단한 웹 스크래퍼를 만들고, 뉴스 기사 10개를 수집하여 CSV로 저장")
result = agent.run()

# 출력 예시:
# 목표: 파이썬으로 간단한 웹 스크래퍼를 만들고...
# 계획 수립 중...
# 총 5개 작업 생성됨
#
# [반복 1] 작업 실행: requests, beautifulsoup4 라이브러리 설치 확인
# [Action] execute_code("import requests; import bs4")
# 결과: 라이브러리 설치 확인 완료
#
# [반복 2] 작업 실행: 뉴스 사이트 HTML 구조 분석
# [Action] search("웹 스크래핑 뉴스 사이트 예시")
# ...
```

### 2. BabyAGI - 작업 관리 Agent

```python
from collections import deque
import openai

class BabyAGI:
    """
    BabyAGI: 작업 생성/우선순위/실행을 자율적으로 관리

    핵심:
    1. Task Creation: 목표 기반 작업 생성
    2. Prioritization: 중요도/의존성 기반 우선순위
    3. Execution: 작업 실행
    4. Loop: 무한 반복
    """
    def __init__(self, objective: str):
        self.objective = objective
        self.task_list = deque()
        self.task_id_counter = 1
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7)

        # 초기 작업 추가
        self.add_task("첫 번째 작업 생성")

    def add_task(self, task_description: str):
        """작업 추가"""
        task = {
            'task_id': self.task_id_counter,
            'task_name': task_description
        }
        self.task_list.append(task)
        self.task_id_counter += 1

    def task_creation_agent(self, result: str, task_description: str):
        """
        작업 생성 Agent

        의도: 현재 결과를 보고 다음 작업들 생성
        """
        prompt = f"""
        목표: {self.objective}

        마지막 완료 작업: {task_description}
        결과: {result}

        현재 작업 목록:
        {json.dumps([t['task_name'] for t in self.task_list], indent=2)}

        목표 달성을 위해 추가로 필요한 작업들을 생성해주세요.
        중복되지 않도록 주의하세요.

        JSON 형식:
        {{
            "new_tasks": ["작업1", "작업2", ...]
        }}
        """

        response = self.llm.invoke(prompt)
        new_tasks_dict = json.loads(response.content)

        for task in new_tasks_dict.get('new_tasks', []):
            self.add_task(task)

    def prioritization_agent(self):
        """
        우선순위 Agent

        의도: 작업 목록을 중요도 순으로 재정렬
        """
        if not self.task_list:
            return

        prompt = f"""
        목표: {self.objective}

        현재 작업 목록:
        {json.dumps([{'id': t['task_id'], 'name': t['task_name']} for t in self.task_list], indent=2)}

        이 작업들을 목표 달성을 위한 중요도 순으로 정렬해주세요.

        JSON 형식:
        {{
            "prioritized_task_ids": [3, 1, 5, 2, ...]
        }}
        """

        response = self.llm.invoke(prompt)
        prioritized = json.loads(response.content)

        # 재정렬
        task_dict = {t['task_id']: t for t in self.task_list}
        self.task_list = deque([
            task_dict[tid] for tid in prioritized['prioritized_task_ids']
            if tid in task_dict
        ])

    def execution_agent(self, task_name: str) -> str:
        """
        실행 Agent

        의도: 작업 수행 및 결과 반환
        """
        prompt = f"""
        목표: {self.objective}
        작업: {task_name}

        이 작업을 수행하고 결과를 상세히 설명해주세요.
        """

        response = self.llm.invoke(prompt)
        return response.content

    def run(self, max_iterations=10):
        """메인 루프"""
        for i in range(max_iterations):
            # Step 1: 우선순위 설정
            self.prioritization_agent()

            # Step 2: 작업 가져오기
            if not self.task_list:
                print("작업 목록이 비어있습니다. 목표 달성!")
                break

            task = self.task_list.popleft()
            print(f"\n[반복 {i+1}] 실행 중: {task['task_name']}")

            # Step 3: 작업 실행
            result = self.execution_agent(task['task_name'])
            print(f"결과: {result[:200]}...")

            # Step 4: 새 작업 생성
            self.task_creation_agent(result, task['task_name'])
            print(f"새 작업 생성 완료. 현재 대기 중인 작업: {len(self.task_list)}개")

# 실전 사용
baby_agi = BabyAGI(objective="AI 기술 블로그 글 작성: Transformer 아키텍처 설명")
baby_agi.run(max_iterations=5)

# 출력:
# [반복 1] 실행 중: 첫 번째 작업 생성
# 결과: Transformer 아키텍처에 대한 개요 리서치를 시작합니다...
# 새 작업 생성 완료. 현재 대기 중인 작업: 4개
#
# [반복 2] 실행 중: Self-Attention 메커니즘 설명 작성
# ...
```

---

## CrewAI - Multi-Agent Collaboration

### 1. 기본 Crew 구성

```python
from crewai import Agent, Task, Crew, Process

class ContentCreationCrew:
    """
    CrewAI: 여러 Agent가 협업하여 작업 수행

    구조:
    - Agent: 역할(role)과 목표(goal)를 가진 개별 에이전트
    - Task: 수행할 작업
    - Crew: Agent + Task 조합, 실행 프로세스 정의
    """
    def __init__(self):
        # Agent 1: 리서처
        self.researcher = Agent(
            role='Senior Researcher',
            goal='최신 AI 트렌드를 조사하고 정확한 정보를 수집',
            backstory="""
            당신은 10년 경력의 AI 연구원입니다.
            최신 논문과 기술 동향에 정통하며, 신뢰할 수 있는 정보만 제공합니다.
            """,
            verbose=True,
            allow_delegation=False,
            tools=[self.search_tool(), self.arxiv_tool()]
        )

        # Agent 2: 작가
        self.writer = Agent(
            role='Tech Writer',
            goal='복잡한 기술을 쉽고 매력적으로 설명',
            backstory="""
            당신은 기술 블로그 전문 작가입니다.
            복잡한 개념을 일반인도 이해할 수 있게 설명하는 데 탁월합니다.
            """,
            verbose=True,
            allow_delegation=False
        )

        # Agent 3: 에디터
        self.editor = Agent(
            role='Editor',
            goal='콘텐츠 품질 검토 및 개선',
            backstory="""
            당신은 까다로운 에디터입니다.
            문법, 논리, 가독성을 꼼꼼히 검토하고 개선점을 제시합니다.
            """,
            verbose=True,
            allow_delegation=True
        )

    def search_tool(self):
        """웹 검색 도구"""
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(max_results=5)

    def arxiv_tool(self):
        """ArXiv 논문 검색"""
        from langchain_community.tools import ArxivQueryRun
        return ArxivQueryRun()

    def create_article(self, topic: str):
        """
        블로그 글 작성 워크플로우
        """
        # Task 1: 리서치
        research_task = Task(
            description=f"""
            주제: {topic}

            다음을 조사하세요:
            1. 최신 동향 및 트렌드
            2. 핵심 기술 및 개념
            3. 실제 응용 사례
            4. 참고 자료 (논문, 블로그 등)

            상세한 리서치 보고서를 작성하세요.
            """,
            agent=self.researcher,
            expected_output="상세한 리서치 보고서 (최소 500단어)"
        )

        # Task 2: 초안 작성
        writing_task = Task(
            description=f"""
            리서치 결과를 바탕으로 블로그 글 초안을 작성하세요.

            구조:
            1. 도입부: 주제 소개 및 중요성
            2. 본문: 핵심 개념 설명 (예시 포함)
            3. 응용: 실제 사용 사례
            4. 결론: 요약 및 향후 전망

            톤: 전문적이지만 친근하게
            길이: 1000-1500 단어
            """,
            agent=self.writer,
            expected_output="완성된 블로그 글 초안",
            context=[research_task]  # 리서치 결과 참조
        )

        # Task 3: 편집
        editing_task = Task(
            description="""
            초안을 검토하고 개선하세요.

            확인 사항:
            1. 문법 및 맞춤법
            2. 논리적 흐름
            3. 가독성
            4. 기술 정확성
            5. 예시의 적절성

            필요시 수정 사항을 적용하고 최종본을 작성하세요.
            """,
            agent=self.editor,
            expected_output="최종 편집된 블로그 글",
            context=[writing_task]  # 초안 참조
        )

        # Crew 생성
        crew = Crew(
            agents=[self.researcher, self.writer, self.editor],
            tasks=[research_task, writing_task, editing_task],
            process=Process.sequential,  # 순차 실행
            verbose=True
        )

        # 실행
        result = crew.kickoff()
        return result

# 실전 사용
crew = ContentCreationCrew()
article = crew.create_article("Vision Transformers (ViT)의 원리와 응용")

# 출력:
# [Researcher] 작업 시작: Vision Transformers 조사...
# [Researcher] ArXiv에서 관련 논문 검색 중...
# [Researcher] 리서치 완료: ViT는 2020년 Google이 발표...
#
# [Writer] 작업 시작: 블로그 글 초안 작성...
# [Writer] 초안 완료: "Vision Transformers: 이미지 인식의 새로운 패러다임"
#
# [Editor] 작업 시작: 초안 검토 중...
# [Editor] 수정 사항: 1) 도입부에 ViT의 중요성 강조, 2) 예시 추가...
# [Editor] 최종 편집 완료!

print(article)
```

### 2. 소프트웨어 개발 Crew

```python
class SoftwareDevelopmentCrew:
    """
    소프트웨어 개발 팀 시뮬레이션

    Agent:
    - Product Manager: 요구사항 정의
    - Architect: 시스템 설계
    - Developer: 코드 작성
    - QA Engineer: 테스트
    """
    def __init__(self):
        # PM
        self.pm = Agent(
            role='Product Manager',
            goal='명확한 요구사항 및 스펙 작성',
            backstory='사용자 중심의 제품을 만드는 PM',
            verbose=True
        )

        # 아키텍트
        self.architect = Agent(
            role='Software Architect',
            goal='확장 가능하고 유지보수 가능한 아키텍처 설계',
            backstory='20년 경력의 시스템 아키텍트',
            verbose=True
        )

        # 개발자
        self.developer = Agent(
            role='Senior Developer',
            goal='깨끗하고 테스트 가능한 코드 작성',
            backstory='Python 전문 개발자',
            verbose=True,
            tools=[self.code_execution_tool()]
        )

        # QA
        self.qa = Agent(
            role='QA Engineer',
            goal='버그 발견 및 품질 보증',
            backstory='꼼꼼한 QA 엔지니어',
            verbose=True
        )

    def code_execution_tool(self):
        """코드 실행 도구"""
        from langchain.tools import Tool

        def execute_python(code: str) -> str:
            try:
                # 안전한 실행 환경 (제한적)
                local_vars = {}
                exec(code, {"__builtins__": {}}, local_vars)
                return str(local_vars)
            except Exception as e:
                return f"오류: {str(e)}"

        return Tool(
            name="PythonExecutor",
            func=execute_python,
            description="Python 코드를 실행하고 결과를 반환"
        )

    def develop_feature(self, feature_request: str):
        """
        기능 개발 워크플로우
        """
        # Task 1: 요구사항 정의
        requirements_task = Task(
            description=f"""
            기능 요청: {feature_request}

            다음을 작성하세요:
            1. 기능 설명
            2. 사용자 스토리
            3. 수용 기준 (Acceptance Criteria)
            4. 제약 사항
            """,
            agent=self.pm,
            expected_output="상세 요구사항 문서"
        )

        # Task 2: 아키텍처 설계
        design_task = Task(
            description="""
            요구사항을 바탕으로 시스템을 설계하세요.

            포함 사항:
            1. 컴포넌트 구조
            2. 데이터 모델
            3. API 인터페이스
            4. 기술 스택 선택
            """,
            agent=self.architect,
            expected_output="아키텍처 설계 문서",
            context=[requirements_task]
        )

        # Task 3: 구현
        implementation_task = Task(
            description="""
            설계를 바탕으로 코드를 작성하세요.

            요구사항:
            1. Python으로 구현
            2. Docstring 포함
            3. Type hints 사용
            4. 단위 테스트 작성

            완성된 코드를 반환하세요.
            """,
            agent=self.developer,
            expected_output="완성된 Python 코드",
            context=[design_task]
        )

        # Task 4: 테스트
        testing_task = Task(
            description="""
            구현된 코드를 테스트하세요.

            테스트 항목:
            1. 기능 동작 확인
            2. Edge case 검증
            3. 성능 체크
            4. 코드 품질 리뷰

            발견된 버그 및 개선 사항을 보고하세요.
            """,
            agent=self.qa,
            expected_output="테스트 보고서",
            context=[implementation_task]
        )

        # Crew 실행
        crew = Crew(
            agents=[self.pm, self.architect, self.developer, self.qa],
            tasks=[requirements_task, design_task, implementation_task, testing_task],
            process=Process.sequential
        )

        result = crew.kickoff()
        return result

# 실전 사용
dev_crew = SoftwareDevelopmentCrew()
result = dev_crew.develop_feature("사용자 인증 시스템 (JWT 기반)")

# 출력:
# [PM] 요구사항 작성 중...
# [PM] 완료: 사용자 스토리 5개 작성 완료
#
# [Architect] 아키텍처 설계 중...
# [Architect] FastAPI + JWT + PostgreSQL 스택 선택
#
# [Developer] 코드 작성 중...
# [Developer] auth.py, models.py, schemas.py 작성 완료
#
# [QA] 테스트 시작...
# [QA] 버그 발견: 토큰 만료 처리 미흡
# [QA] 보고서 작성 완료
```

---

## 고급 패턴

### 1. LangGraph - 조건부 워크플로우

```python
from langgraph.graph import Graph, END
from typing import TypedDict

class ResearchState(TypedDict):
    """상태 정의"""
    query: str
    search_results: list
    analysis: str
    report: str
    needs_more_info: bool

class ResearchAgent:
    """
    LangGraph: 조건부 흐름 제어

    노드: 각 처리 단계
    엣지: 조건부 전환
    """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    def search_node(self, state: ResearchState) -> ResearchState:
        """검색 노드"""
        from langchain_community.tools.tavily_search import TavilySearchResults

        search = TavilySearchResults(max_results=5)
        results = search.invoke(state['query'])

        state['search_results'] = results
        return state

    def analyze_node(self, state: ResearchState) -> ResearchState:
        """분석 노드"""
        prompt = f"""
        질문: {state['query']}

        검색 결과:
        {json.dumps(state['search_results'], indent=2)}

        이 정보가 질문에 충분히 답변하는지 분석하고,
        부족하다면 어떤 정보가 더 필요한지 설명하세요.

        JSON 형식:
        {{
            "is_sufficient": true/false,
            "analysis": "...",
            "missing_info": "..."
        }}
        """

        response = self.llm.invoke(prompt)
        analysis = json.loads(response.content)

        state['analysis'] = analysis['analysis']
        state['needs_more_info'] = not analysis['is_sufficient']

        if not analysis['is_sufficient']:
            # 추가 검색 쿼리 생성
            state['query'] = analysis.get('missing_info', state['query'])

        return state

    def report_node(self, state: ResearchState) -> ResearchState:
        """보고서 작성 노드"""
        prompt = f"""
        질문: {state['query']}

        검색 결과:
        {json.dumps(state['search_results'], indent=2)}

        분석:
        {state['analysis']}

        종합적인 보고서를 작성하세요.
        """

        response = self.llm.invoke(prompt)
        state['report'] = response.content
        return state

    def should_continue(self, state: ResearchState) -> str:
        """조건부 라우팅"""
        if state.get('needs_more_info', False):
            return "search"  # 다시 검색
        else:
            return "report"  # 보고서 작성

    def create_graph(self):
        """그래프 생성"""
        workflow = Graph()

        # 노드 추가
        workflow.add_node("search", self.search_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("report", self.report_node)

        # 엣지 추가
        workflow.set_entry_point("search")
        workflow.add_edge("search", "analyze")

        # 조건부 엣지
        workflow.add_conditional_edges(
            "analyze",
            self.should_continue,
            {
                "search": "search",  # 정보 부족 → 재검색
                "report": "report"   # 충분 → 보고서 작성
            }
        )

        workflow.add_edge("report", END)

        return workflow.compile()

    def research(self, query: str):
        """리서치 실행"""
        graph = self.create_graph()

        initial_state = ResearchState(
            query=query,
            search_results=[],
            analysis="",
            report="",
            needs_more_info=True
        )

        result = graph.invoke(initial_state)
        return result['report']

# 실전 사용
researcher = ResearchAgent()
report = researcher.research("Mixture of Experts (MoE)의 최신 동향은?")

print(report)
# 출력:
# Mixture of Experts (MoE) 동향 보고서
#
# 1. 개요
# MoE는 2024년 가장 주목받는 아키텍처 중 하나로...
#
# 2. 주요 모델
# - Mixtral 8x7B (Mistral AI): 7B 전문가 8개...
# - GPT-4: MoE 구조로 추정...
#
# 3. 장점
# - 계산 효율: 선택적 활성화...
```

### 2. Human-in-the-Loop

```python
class HumanInLoopAgent:
    """
    Human-in-the-Loop: 중요한 결정에 사람 개입

    사용 사례:
    - 비용이 큰 작업 (API 호출, 결제)
    - 중요한 결정 (데이터 삭제, 배포)
    - 품질 검증 (최종 리뷰)
    """
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    def get_human_feedback(self, question: str, options: list = None) -> str:
        """
        사람에게 피드백 요청
        """
        print(f"\n{'='*50}")
        print(f"질문: {question}")

        if options:
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")
            choice = int(input("선택 (숫자): ")) - 1
            return options[choice]
        else:
            return input("답변: ")

    def execute_with_approval(self, task: str, action_plan: dict):
        """
        승인 후 실행
        """
        print(f"\n작업: {task}")
        print(f"계획:")
        print(json.dumps(action_plan, indent=2, ensure_ascii=False))

        approval = self.get_human_feedback(
            "이 계획을 실행하시겠습니까?",
            options=["예", "아니오", "수정 후 재시도"]
        )

        if approval == "예":
            # 실행
            print("실행 중...")
            return self.execute_action(action_plan)
        elif approval == "수정 후 재시도":
            # 사용자 입력 받아 계획 수정
            feedback = self.get_human_feedback("어떻게 수정할까요?")

            # LLM에게 계획 재수립 요청
            revised_plan = self.revise_plan(action_plan, feedback)
            return self.execute_with_approval(task, revised_plan)
        else:
            print("작업 취소")
            return None

    def revise_plan(self, original_plan: dict, feedback: str) -> dict:
        """계획 수정"""
        prompt = f"""
        원래 계획:
        {json.dumps(original_plan, indent=2)}

        피드백:
        {feedback}

        피드백을 반영하여 계획을 수정하세요.
        JSON 형식으로 반환.
        """

        response = self.llm.invoke(prompt)
        return json.loads(response.content)

    def execute_action(self, action_plan: dict):
        """액션 실행 (예시)"""
        # 실제 구현
        return "작업 완료"

# 실전 사용
agent = HumanInLoopAgent()

task = "데이터베이스에서 6개월 이상 비활성 사용자 삭제"
plan = {
    "action": "delete_users",
    "criteria": "last_login < 6 months ago",
    "estimated_count": 1523
}

result = agent.execute_with_approval(task, plan)

# 출력:
# 작업: 데이터베이스에서 6개월 이상 비활성 사용자 삭제
# 계획:
# {
#   "action": "delete_users",
#   "criteria": "last_login < 6 months ago",
#   "estimated_count": 1523
# }
#
# 질문: 이 계획을 실행하시겠습니까?
# 1. 예
# 2. 아니오
# 3. 수정 후 재시도
# 선택 (숫자): 3
#
# 답변: 1년 이상 비활성 사용자만 삭제해주세요
# [계획 재수립...]
```

---

## 프로덕션 배포

### 1. Agent 모니터링

```python
from prometheus_client import Counter, Histogram, Gauge
import time
import logging

class AgentMonitoring:
    """
    Agent 모니터링 시스템

    추적 메트릭:
    1. 작업 성공률
    2. 평균 반복 횟수
    3. 도구 사용 빈도
    4. 비용
    """
    def __init__(self):
        # 메트릭 정의
        self.task_counter = Counter(
            'agent_tasks_total',
            'Total agent tasks',
            ['status', 'agent_type']
        )

        self.iteration_counter = Histogram(
            'agent_iterations',
            'Number of iterations per task',
            ['agent_type']
        )

        self.tool_usage = Counter(
            'agent_tool_usage_total',
            'Tool usage count',
            ['tool_name', 'agent_type']
        )

        self.cost_counter = Counter(
            'agent_cost_dollars',
            'Total cost in dollars',
            ['agent_type']
        )

        self.active_agents = Gauge(
            'agent_active_count',
            'Number of active agents',
            ['agent_type']
        )

    def track_task(self, agent_type: str, status: str, iterations: int, cost: float):
        """작업 추적"""
        self.task_counter.labels(status=status, agent_type=agent_type).inc()
        self.iteration_counter.labels(agent_type=agent_type).observe(iterations)
        self.cost_counter.labels(agent_type=agent_type).inc(cost)

    def track_tool_use(self, agent_type: str, tool_name: str):
        """도구 사용 추적"""
        self.tool_usage.labels(tool_name=tool_name, agent_type=agent_type).inc()

    def get_stats(self):
        """통계 조회"""
        # Prometheus에서 조회하거나 직접 계산
        pass

# 실전 사용 - Agent Wrapper
class MonitoredAgent:
    """모니터링이 포함된 Agent"""
    def __init__(self, agent, agent_type: str, monitoring: AgentMonitoring):
        self.agent = agent
        self.agent_type = agent_type
        self.monitoring = monitoring

    def run(self, task: str):
        """모니터링과 함께 실행"""
        start_time = time.time()
        iterations = 0
        cost = 0.0

        try:
            # Active agent 증가
            self.monitoring.active_agents.labels(agent_type=self.agent_type).inc()

            # Agent 실행
            result = self.agent.run(task)

            # 성공 기록
            iterations = getattr(self.agent, 'iteration_count', 1)
            cost = getattr(self.agent, 'total_cost', 0.01)

            self.monitoring.track_task(
                self.agent_type,
                status='success',
                iterations=iterations,
                cost=cost
            )

            return result

        except Exception as e:
            # 실패 기록
            logging.error(f"Agent 오류: {str(e)}")
            self.monitoring.track_task(
                self.agent_type,
                status='failure',
                iterations=iterations,
                cost=cost
            )
            raise

        finally:
            # Active agent 감소
            self.monitoring.active_agents.labels(agent_type=self.agent_type).dec()

            duration = time.time() - start_time
            logging.info(f"Agent 실행 완료: {duration:.2f}초, {iterations}회 반복, ${cost:.4f}")

# 사용
monitoring = AgentMonitoring()
base_agent = BasicAgent()
monitored_agent = MonitoredAgent(base_agent, "react_agent", monitoring)

result = monitored_agent.run("파이썬으로 피보나치 수열 코드를 작성해줘")
```

### 2. 에러 핸들링 & 재시도

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import functools

class RobustAgent:
    """
    견고한 Agent: 에러 핸들링, 재시도, Fallback

    전략:
    1. 자동 재시도 (exponential backoff)
    2. Fallback (간단한 모델로 전환)
    3. Circuit Breaker (연속 실패 시 중단)
    """
    def __init__(self):
        self.llm_primary = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        self.llm_fallback = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        self.failure_count = 0
        self.circuit_breaker_threshold = 5

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def call_llm_with_retry(self, prompt: str):
        """
        재시도 로직

        - 1차 시도: 즉시
        - 2차 시도: 2초 대기
        - 3차 시도: 4초 대기
        """
        try:
            response = self.llm_primary.invoke(prompt)
            self.failure_count = 0  # 성공 시 리셋
            return response.content

        except Exception as e:
            self.failure_count += 1
            logging.warning(f"LLM 호출 실패 ({self.failure_count}회): {str(e)}")

            # Circuit Breaker
            if self.failure_count >= self.circuit_breaker_threshold:
                raise Exception("Circuit Breaker 작동: 너무 많은 실패")

            raise  # 재시도를 위해 예외 전파

    def call_with_fallback(self, prompt: str):
        """
        Fallback 전략

        Primary 실패 시 → Fallback 모델 사용
        """
        try:
            return self.call_llm_with_retry(prompt)
        except Exception as e:
            logging.warning(f"Primary 모델 실패, Fallback 사용: {str(e)}")
            try:
                response = self.llm_fallback.invoke(prompt)
                return response.content
            except Exception as e2:
                logging.error(f"Fallback도 실패: {str(e2)}")
                return "죄송합니다. 현재 서비스를 이용할 수 없습니다."

    def run_safe(self, task: str):
        """안전한 실행"""
        try:
            result = self.call_with_fallback(f"다음 작업을 수행하세요: {task}")
            return {
                'success': True,
                'result': result
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# 실전 사용
agent = RobustAgent()
result = agent.run_safe("복잡한 수학 문제 풀기")

if result['success']:
    print(result['result'])
else:
    print(f"오류: {result['error']}")
```

---

## 학습 로드맵

### 기초 (1-2주)
1. **LangChain 기초**
   - ReAct Agent 구현
   - Custom Tools 작성
   - Memory 이해

2. **도구 통합**
   - 웹 검색 (Tavily, Google)
   - 계산기, 날씨 API
   - 데이터베이스 연결

### 중급 (2-3주)
3. **AutoGPT 패턴**
   - 작업 분해 (Planning)
   - Self-Reflection
   - 파일 시스템 접근

4. **Multi-Agent**
   - CrewAI로 협업 시스템
   - 역할 분담 전략
   - 워크플로우 설계

### 고급 (3-4주)
5. **LangGraph**
   - 조건부 흐름
   - 복잡한 워크플로우
   - 상태 관리

6. **프로덕션**
   - 모니터링, 에러 핸들링
   - 비용 최적화
   - Human-in-the-Loop

---

## 핵심 요약

### 주요 프레임워크
- **LangChain**: 범용성, ReAct 패턴
- **AutoGPT**: 자율성, 장기 목표
- **CrewAI**: 멀티 에이전트 협업
- **LangGraph**: 복잡한 워크플로우

### 핵심 패턴
1. **ReAct**: Thought → Action → Observation
2. **Planning**: 목표 → 하위 작업 분해
3. **Reflection**: 실수 인지 및 개선
4. **Collaboration**: 역할 분담 협업

### 프로덕션 고려사항
- **모니터링**: 성공률, 비용 추적
- **견고성**: 재시도, Fallback, Circuit Breaker
- **Human-in-the-Loop**: 중요한 결정에 사람 개입

### 다음 단계
- [LLMOps](./04-llmops-production.md) - 프로덕션 배포
- [Multimodal AI](./05-multimodal-ai.md) - 이미지/비디오 Agent
- [Research Trends](./02-research-trends-2024-2025.md) - Multi-Agent 최신 연구

---

**작성일**: 2024-11-18
**업데이트**: Phase 6 - Current Trends
