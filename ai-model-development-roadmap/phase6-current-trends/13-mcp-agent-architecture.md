# MCP + AI Agent 아키텍처: 실전 구현 가이드

## 핵심 개념

**"AI Agent + MCP = 마이크로서비스처럼 구성 가능한 AI 시스템"**

```python
# 의도: 여러 MCP를 조합해서 복잡한 작업 자동화
# 아이디어: 각 MCP가 특정 기능을 담당하고, Agent가 orchestration

"""
전통적인 AI:
- LLM 하나로 모든 것 해결 시도
- 환각(hallucination) 문제
- 실시간 데이터 접근 불가
- 외부 시스템 연동 어려움

MCP + Agent 방식:
- Agent: 사고하고 계획 (LLM)
- MCP 서버들: 실제 작업 실행 (Tools)
- 각 MCP는 독립적 기능 제공
- Agent가 MCP들을 조합

비유:
- Agent = 오케스트라 지휘자
- MCP 서버들 = 각 악기 연주자
- 결과 = 아름다운 하모니
"""
```

---

## MCP (Model Context Protocol)란?

### 기본 구조

```python
"""
┌──────────────────────────────────────────────────────┐
│                    AI Agent                          │
│            (Claude, GPT-4, etc.)                     │
│                                                      │
│  "사용자 요청 분석 → 계획 수립 → MCP 호출 → 결과 종합" │
└──────────────────────────────────────────────────────┘
           │           │           │           │
           ▼           ▼           ▼           ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
     │Database │ │   Web   │ │  File   │ │  Email  │
     │   MCP   │ │   MCP   │ │   MCP   │ │   MCP   │
     └─────────┘ └─────────┘ └─────────┘ └─────────┘
           │           │           │           │
           ▼           ▼           ▼           ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
     │Postgres │ │ Brave   │ │ Local   │ │  SMTP   │
     │         │ │ Search  │ │  Disk   │ │ Server  │
     └─────────┘ └─────────┘ └─────────┘ └─────────┘

각 MCP 서버:
- 독립적으로 실행
- 표준 프로토콜 (MCP) 구현
- Tools 제공 (list_tools, call_tool)
- Resources 제공 (파일, 데이터)
- Prompts 제공 (템플릿)
"""
```

### MCP 서버 구현 예시

```python
# 의도: 가장 기본적인 MCP 서버
# 아이디어: Database 연동 MCP

from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncpg
from typing import Dict, List

class DatabaseMCP(Server):
    """PostgreSQL 데이터베이스 MCP 서버"""

    def __init__(self):
        super().__init__("database-mcp")
        self.pool = None

    async def initialize(self):
        """DB 연결 풀 생성"""
        self.pool = await asyncpg.create_pool(
            host='localhost',
            database='mydb',
            user='user',
            password='password',
            min_size=5,
            max_size=20
        )

    async def list_tools(self) -> List[Tool]:
        """
        Agent가 사용 가능한 도구 목록 반환
        - Agent는 이 목록을 보고 어떤 작업을 할 수 있는지 파악
        """
        return [
            Tool(
                name="query_database",
                description="SQL 쿼리 실행. SELECT 문만 허용. 매개변수: query (SQL), params (바인딩 값)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "실행할 SQL SELECT 쿼리"
                        },
                        "params": {
                            "type": "array",
                            "description": "SQL 파라미터 (선택)",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_schema",
                description="테이블 스키마 조회. 매개변수: table_name",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "조회할 테이블 이름"
                        }
                    },
                    "required": ["table_name"]
                }
            ),
            Tool(
                name="list_tables",
                description="데이터베이스의 모든 테이블 목록 조회. 매개변수 없음",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        """
        Agent가 도구를 호출할 때 실행되는 함수

        매개변수:
        - name: 도구 이름 (예: "query_database")
        - arguments: 도구 매개변수 (예: {"query": "SELECT * FROM users"})

        반환:
        - 도구 실행 결과 (TextContent)
        """
        if name == "query_database":
            query = arguments['query']
            params = arguments.get('params', [])

            # 보안: SELECT만 허용
            if not query.strip().upper().startswith('SELECT'):
                return [TextContent(
                    type="text",
                    text="Error: SELECT 쿼리만 허용됩니다."
                )]

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

                # 결과를 Markdown 테이블로 포맷
                if not rows:
                    return [TextContent(
                        type="text",
                        text="결과가 없습니다."
                    )]

                # 컬럼 이름
                columns = list(rows[0].keys())

                # Markdown 테이블 생성
                table = "| " + " | ".join(columns) + " |\n"
                table += "| " + " | ".join(["---"] * len(columns)) + " |\n"

                for row in rows[:100]:  # 최대 100개
                    values = [str(row[col]) for col in columns]
                    table += "| " + " | ".join(values) + " |\n"

                result_text = f"쿼리 실행 성공. {len(rows)}개 행 반환:\n\n{table}"

                if len(rows) > 100:
                    result_text += f"\n... 그 외 {len(rows) - 100}개 행"

                return [TextContent(type="text", text=result_text)]

        elif name == "get_schema":
            table_name = arguments['table_name']

            query = """
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY ordinal_position
            """

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, table_name)

                if not rows:
                    return [TextContent(
                        type="text",
                        text=f"테이블 '{table_name}'을 찾을 수 없습니다."
                    )]

                schema_text = f"테이블 '{table_name}' 스키마:\n\n"
                schema_text += "| 컬럼명 | 타입 | Nullable | 기본값 |\n"
                schema_text += "| --- | --- | --- | --- |\n"

                for row in rows:
                    schema_text += f"| {row['column_name']} | {row['data_type']} | {row['is_nullable']} | {row['column_default'] or 'N/A'} |\n"

                return [TextContent(type="text", text=schema_text)]

        elif name == "list_tables":
            query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query)

                table_list = "\n".join([f"- {row['table_name']}" for row in rows])
                result_text = f"데이터베이스 테이블 목록 ({len(rows)}개):\n\n{table_list}"

                return [TextContent(type="text", text=result_text)]

# MCP 서버 실행
async def main():
    server = DatabaseMCP()
    await server.initialize()
    await server.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## 실전 유즈케이스 1: 고객 지원 자동화 (Customer Support Automation)

```python
"""
시나리오: "고객 john@example.com의 최근 주문 상태를 확인하고 배송 지연 사유를 이메일로 보내줘"

필요한 MCP:
1. Database MCP - 고객 정보, 주문 정보 조회
2. Email MCP - 이메일 발송
3. CRM MCP - 고객 interaction 기록
"""

# ============= Email MCP =============
class EmailMCP(Server):
    """이메일 발송 MCP 서버"""

    def __init__(self):
        super().__init__("email-mcp")
        self.smtp_host = "smtp.gmail.com"
        self.smtp_port = 587

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="send_email",
                description="이메일 발송. 매개변수: to, subject, body, cc (선택), bcc (선택)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "to": {
                            "type": "string",
                            "description": "수신자 이메일"
                        },
                        "subject": {
                            "type": "string",
                            "description": "제목"
                        },
                        "body": {
                            "type": "string",
                            "description": "본문 (HTML 또는 텍스트)"
                        },
                        "cc": {
                            "type": "array",
                            "description": "참조 (선택)",
                            "items": {"type": "string"}
                        },
                        "template": {
                            "type": "string",
                            "description": "템플릿 이름 (선택)"
                        }
                    },
                    "required": ["to", "subject", "body"]
                }
            ),
            Tool(
                name="get_email_templates",
                description="이메일 템플릿 목록 조회. 매개변수 없음",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "send_email":
            # SMTP로 이메일 발송
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart('alternative')
            msg['Subject'] = arguments['subject']
            msg['From'] = "support@company.com"
            msg['To'] = arguments['to']

            if arguments.get('cc'):
                msg['Cc'] = ", ".join(arguments['cc'])

            # HTML 본문
            html_part = MIMEText(arguments['body'], 'html')
            msg.attach(html_part)

            # SMTP 발송
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login("support@company.com", "password")
                server.send_message(msg)

            return [TextContent(
                type="text",
                text=f"이메일 발송 완료\n받는 사람: {arguments['to']}\n제목: {arguments['subject']}"
            )]


# ============= CRM MCP =============
class CRMMCP(Server):
    """CRM (Salesforce, HubSpot 등) MCP 서버"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="create_interaction",
                description="고객 interaction 기록 생성. 매개변수: customer_email, type, notes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "customer_email": {"type": "string"},
                        "interaction_type": {
                            "type": "string",
                            "enum": ["email", "call", "chat", "meeting"]
                        },
                        "subject": {"type": "string"},
                        "notes": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["open", "in_progress", "resolved"],
                            "default": "resolved"
                        }
                    },
                    "required": ["customer_email", "interaction_type", "subject", "notes"]
                }
            ),
            Tool(
                name="get_customer_history",
                description="고객 interaction 이력 조회. 매개변수: customer_email, limit",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "customer_email": {"type": "string"},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["customer_email"]
                }
            )
        ]


# ============= Agent 실행 =============
user_request = "고객 john@example.com의 최근 주문 상태를 확인하고 배송 지연 사유를 이메일로 보내줘"

"""
Agent 자동 실행 순서:

STEP 1: query_database (Database MCP)
Agent 생각: "먼저 고객 정보를 찾아야겠다"

SQL:
```sql
SELECT customer_id, name, email
FROM customers
WHERE email = 'john@example.com'
```

결과:
| customer_id | name      | email              |
|-------------|-----------|-------------------|
| 12345       | John Doe  | john@example.com  |


STEP 2: query_database (Database MCP)
Agent 생각: "최근 주문을 찾자"

SQL:
```sql
SELECT order_id, order_date, status, total_amount
FROM orders
WHERE customer_id = 12345
ORDER BY order_date DESC
LIMIT 5
```

결과:
| order_id | order_date | status      | total_amount |
|----------|------------|-------------|--------------|
| ORD-789  | 2024-01-15 | shipped     | $129.99      |
| ORD-745  | 2024-01-10 | delivered   | $45.50       |
| ORD-701  | 2023-12-20 | delivered   | $89.00       |


STEP 3: query_database (Database MCP)
Agent 생각: "최신 주문 ORD-789의 배송 정보를 확인하자"

SQL:
```sql
SELECT s.*, c.name as carrier_name
FROM shipments s
JOIN carriers c ON s.carrier_id = c.carrier_id
WHERE s.order_id = 'ORD-789'
```

결과:
| shipment_id | order_id | carrier_name | tracking_number | status        | estimated_delivery | actual_delivery |
|-------------|----------|--------------|-----------------|---------------|-------------------|-----------------|
| SHIP-456    | ORD-789  | FedEx        | 1234567890      | in_transit    | 2024-01-17        | NULL            |


STEP 4: LLM 분석
Agent 생각: "배송 상태를 분석하자"
- 주문일: 2024-01-15
- 예상 배송일: 2024-01-17
- 오늘: 2024-01-18 (지연!)
- 현재 상태: in_transit

Agent 판단: "배송이 하루 지연되었고, 아직 배송 중이다. 고객에게 사과 이메일을 보내야겠다"


STEP 5: send_email (Email MCP)
Agent가 이메일 작성:

Request:
```json
{
  "to": "john@example.com",
  "subject": "주문 ORD-789 배송 지연 안내",
  "body": "
<html>
<body>
<p>안녕하세요 John Doe 고객님,</p>

<p>주문하신 상품의 배송이 예정보다 하루 지연되고 있어 안내 드립니다.</p>

<h3>주문 정보</h3>
<ul>
  <li>주문 번호: ORD-789</li>
  <li>주문 금액: $129.99</li>
  <li>배송사: FedEx</li>
  <li>운송장 번호: 1234567890</li>
</ul>

<h3>배송 상태</h3>
<ul>
  <li>주문일: 2024-01-15</li>
  <li>예상 배송일: 2024-01-17</li>
  <li><strong>수정 예상 배송일: 2024-01-19</strong></li>
  <li>현재 상태: 배송 중</li>
</ul>

<p>지연 사유: 기상 악화로 인한 운송 지연</p>

<p>불편을 드려 죄송합니다. 빠른 배송을 위해 최선을 다하고 있습니다.</p>

<p>추가 문의사항이 있으시면 언제든 연락 주세요.</p>

<p>감사합니다.<br>
고객지원팀</p>
</body>
</html>
  "
}
```

결과: "이메일 발송 완료"


STEP 6: create_interaction (CRM MCP)
Agent 생각: "CRM에 이 interaction을 기록해야겠다"

Request:
```json
{
  "customer_email": "john@example.com",
  "interaction_type": "email",
  "subject": "주문 ORD-789 배송 지연 안내",
  "notes": "고객에게 배송 지연 사유 안내 이메일 발송. 주문 ORD-789, FedEx 1234567890, 기상 악화로 1일 지연. 수정 예상 배송일: 2024-01-19",
  "status": "resolved"
}
```

결과: "Interaction 기록 완료"


STEP 7: 최종 응답
"John Doe 고객님께 배송 지연 안내 이메일을 발송했습니다.

주문 정보:
- 주문 번호: ORD-789
- 주문 금액: $129.99
- 배송사: FedEx (운송장: 1234567890)

배송 상태:
- 예상 배송일: 2024-01-17 → 2024-01-19 (1일 지연)
- 지연 사유: 기상 악화
- 현재 상태: 배송 중

이메일 내용:
- 지연 사유 설명
- 수정 예상 배송일 안내
- 사과 메시지

CRM에 interaction이 기록되었습니다."

처리 시간: 5초
"""

"""
ROI:

Before (수동):
- CS 담당자가 직접 처리
  1. 고객 이메일 확인
  2. CRM에서 고객 찾기
  3. 주문 시스템에서 주문 찾기
  4. 배송 시스템에서 배송 상태 확인
  5. 이메일 작성
  6. CRM에 기록
- 소요 시간: 10-15분

After (Agent):
- 자연어 요청만으로 자동 처리
- 소요 시간: 5초

하루 100건 처리 시:
- 시간 절감: 25시간 → 8분
- 인건비 절감: $500/day
- 응답 속도: 99% 향상
- 고객 만족도: +40%
"""
```

---

## 실전 유즈케이스 2: 데이터 분석 파이프라인 (Data Analysis Pipeline)

```python
"""
시나리오: "지난 달 매출 데이터를 분석해서 트렌드 그래프와 함께 경영진에게 리포트 보내줘"

필요한 MCP:
1. Database MCP - 매출 데이터 조회
2. Python MCP - 데이터 분석, 그래프 생성
3. File System MCP - 그래프 이미지 저장
4. Email MCP - 리포트 이메일 발송
"""

# ============= Python MCP =============
class PythonMCP(Server):
    """Python 코드 실행 MCP 서버"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="execute_python",
                description="Python 코드 실행. pandas, numpy, matplotlib 사용 가능. 매개변수: code",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "실행할 Python 코드"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "타임아웃 (초), 기본 30초",
                            "default": 30
                        }
                    },
                    "required": ["code"]
                }
            ),
            Tool(
                name="install_package",
                description="Python 패키지 설치 (pip). 매개변수: package",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "package": {"type": "string", "description": "패키지 이름"}
                    },
                    "required": ["package"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "execute_python":
            import subprocess
            import tempfile

            code = arguments['code']
            timeout = arguments.get('timeout', 30)

            # 임시 파일에 코드 작성
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                # Python 실행
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                output = result.stdout
                errors = result.stderr

                if errors:
                    return [TextContent(
                        type="text",
                        text=f"실행 완료 (경고 있음):\n\nOutput:\n{output}\n\nWarnings:\n{errors}"
                    )]
                else:
                    return [TextContent(
                        type="text",
                        text=f"실행 완료:\n\n{output}"
                    )]

            except subprocess.TimeoutExpired:
                return [TextContent(
                    type="text",
                    text=f"Error: 타임아웃 ({timeout}초 초과)"
                )]
            finally:
                import os
                os.unlink(temp_file)


# ============= File System MCP =============
class FileSystemMCP(Server):
    """파일 시스템 MCP 서버"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="read_file",
                description="파일 읽기. 매개변수: path",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "파일 경로"}
                    },
                    "required": ["path"]
                }
            ),
            Tool(
                name="write_file",
                description="파일 쓰기. 매개변수: path, content",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            ),
            Tool(
                name="list_directory",
                description="디렉토리 내용 조회. 매개변수: path",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }
            )
        ]


# ============= Agent 실행 =============
user_request = "지난 달 매출 데이터를 분석해서 트렌드 그래프와 함께 경영진에게 리포트 보내줘"

"""
Agent 자동 실행 순서:

STEP 1: query_database (Database MCP)
Agent 생각: "지난 달(2024년 1월) 매출 데이터를 가져오자"

SQL:
```sql
SELECT
    DATE(order_date) as date,
    COUNT(*) as order_count,
    SUM(total_amount) as daily_revenue,
    AVG(total_amount) as avg_order_value
FROM orders
WHERE order_date >= '2024-01-01'
  AND order_date < '2024-02-01'
  AND status != 'cancelled'
GROUP BY DATE(order_date)
ORDER BY date
```

결과: 31일 치 데이터 (CSV 형식)
```
date,order_count,daily_revenue,avg_order_value
2024-01-01,45,5430.50,120.68
2024-01-02,52,6240.00,120.00
...
2024-01-31,63,7560.00,120.00
```


STEP 2: execute_python (Python MCP)
Agent 생각: "pandas로 데이터 분석하고 matplotlib으로 그래프 생성하자"

Python 코드:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 데이터 로드 (Agent가 이전 단계 결과를 전달)
data = '''
date,order_count,daily_revenue,avg_order_value
2024-01-01,45,5430.50,120.68
2024-01-02,52,6240.00,120.00
...
2024-01-31,63,7560.00,120.00
'''

df = pd.read_csv(pd.io.common.StringIO(data))
df['date'] = pd.to_datetime(df['date'])

# 통계 계산
total_revenue = df['daily_revenue'].sum()
avg_daily_revenue = df['daily_revenue'].mean()
total_orders = df['order_count'].sum()
avg_daily_orders = df['order_count'].mean()

print(f"=== 2024년 1월 매출 분석 ===")
print(f"총 매출: ${total_revenue:,.2f}")
print(f"일평균 매출: ${avg_daily_revenue:,.2f}")
print(f"총 주문: {total_orders:,}건")
print(f"일평균 주문: {avg_daily_orders:.1f}건")
print(f"평균 객단가: ${df['avg_order_value'].mean():.2f}")

# 트렌드 분석
revenue_growth = (df['daily_revenue'].iloc[-7:].mean() - df['daily_revenue'].iloc[:7].mean()) / df['daily_revenue'].iloc[:7].mean() * 100
print(f"\n월초 대비 월말 매출 증가율: {revenue_growth:.1f}%")

# 그래프 생성
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 그래프 1: 일별 매출
axes[0, 0].plot(df['date'], df['daily_revenue'], marker='o', linewidth=2, color='#2E86AB')
axes[0, 0].set_title('일별 매출 트렌드', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('날짜')
axes[0, 0].set_ylabel('매출 ($)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# 그래프 2: 일별 주문 건수
axes[0, 1].bar(df['date'], df['order_count'], color='#A23B72', alpha=0.7)
axes[0, 1].set_title('일별 주문 건수', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('날짜')
axes[0, 1].set_ylabel('주문 건수')
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].tick_params(axis='x', rotation=45)

# 그래프 3: 7일 이동평균
df['revenue_ma7'] = df['daily_revenue'].rolling(window=7).mean()
axes[1, 0].plot(df['date'], df['daily_revenue'], alpha=0.3, label='일별 매출')
axes[1, 0].plot(df['date'], df['revenue_ma7'], linewidth=3, color='#F18F01', label='7일 이동평균')
axes[1, 0].set_title('매출 트렌드 (7일 이동평균)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('날짜')
axes[1, 0].set_ylabel('매출 ($)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].tick_params(axis='x', rotation=45)

# 그래프 4: 주별 비교
df['week'] = df['date'].dt.isocalendar().week
weekly = df.groupby('week').agg({
    'daily_revenue': 'sum',
    'order_count': 'sum'
})
axes[1, 1].bar(weekly.index, weekly['daily_revenue'], color='#6A994E', alpha=0.7)
axes[1, 1].set_title('주별 매출', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('주차')
axes[1, 1].set_ylabel('매출 ($)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/tmp/sales_report_2024_01.png', dpi=300, bbox_inches='tight')
print("\n그래프 저장 완료: /tmp/sales_report_2024_01.png")
```

실행 결과:
```
=== 2024년 1월 매출 분석 ===
총 매출: $186,432.50
일평균 매출: $6,014.27
총 주문: 1,547건
일평균 주문: 49.9건
평균 객단가: $120.52

월초 대비 월말 매출 증가율: 15.3%

그래프 저장 완료: /tmp/sales_report_2024_01.png
```


STEP 3: read_file (File System MCP)
Agent 생각: "그래프 이미지를 읽어서 이메일에 첨부하자"

Request: {"path": "/tmp/sales_report_2024_01.png"}

결과: [이미지 바이너리 데이터]


STEP 4: send_email (Email MCP)
Agent가 종합 리포트 이메일 작성:

Request:
```json
{
  "to": "ceo@company.com",
  "cc": ["cfo@company.com", "cmo@company.com"],
  "subject": "[월간 리포트] 2024년 1월 매출 분석",
  "body": "
<html>
<body style='font-family: Arial, sans-serif;'>
<h2>2024년 1월 매출 분석 리포트</h2>

<h3>📊 핵심 지표</h3>
<table border='1' cellpadding='10' style='border-collapse: collapse;'>
  <tr style='background-color: #f0f0f0;'>
    <th>지표</th>
    <th>값</th>
  </tr>
  <tr>
    <td>총 매출</td>
    <td><strong>$186,432.50</strong></td>
  </tr>
  <tr>
    <td>총 주문 건수</td>
    <td><strong>1,547건</strong></td>
  </tr>
  <tr>
    <td>일평균 매출</td>
    <td>$6,014.27</td>
  </tr>
  <tr>
    <td>일평균 주문</td>
    <td>49.9건</td>
  </tr>
  <tr>
    <td>평균 객단가</td>
    <td>$120.52</td>
  </tr>
</table>

<h3>📈 주요 인사이트</h3>
<ul>
  <li><strong>매출 증가 트렌드</strong>: 월초 대비 월말 매출 15.3% 증가</li>
  <li><strong>안정적 성장</strong>: 7일 이동평균 우상향</li>
  <li><strong>객단가 유지</strong>: $120 수준 안정적 유지</li>
</ul>

<h3>🎯 권장 사항</h3>
<ol>
  <li>성장세 유지를 위한 마케팅 예산 확대</li>
  <li>평균 객단가 상향을 위한 크로스셀링 전략</li>
  <li>주말 매출 강화 프로모션 검토</li>
</ol>

<h3>📉 첨부 그래프</h3>
<img src='cid:sales_chart' alt='매출 트렌드' style='max-width: 100%; height: auto;' />

<hr>
<p style='color: #666; font-size: 12px;'>
본 리포트는 AI Agent가 자동 생성했습니다.<br>
생성 시각: 2024-02-01 09:30:00<br>
문의: data-team@company.com
</p>
</body>
</html>
  ",
  "attachments": [
    {
      "filename": "sales_report_2024_01.png",
      "content": "[이미지 데이터]",
      "cid": "sales_chart"
    }
  ]
}
```

결과: "이메일 발송 완료"


STEP 5: 최종 응답
"2024년 1월 매출 분석 리포트를 경영진에게 발송했습니다.

핵심 지표:
- 총 매출: $186,432.50
- 총 주문: 1,547건
- 월초 대비 월말 매출 증가율: 15.3%

발송 대상:
- CEO (수신)
- CFO, CMO (참조)

첨부:
- 매출 트렌드 그래프 (4개 차트)
  1. 일별 매출 트렌드
  2. 일별 주문 건수
  3. 7일 이동평균
  4. 주별 비교

권장 사항 3가지 포함되어 있습니다."

처리 시간: 8분
"""

"""
ROI:

Before (수동):
- 데이터 분석가가 처리
  1. SQL 쿼리 작성 (30분)
  2. 데이터 다운로드 (5분)
  3. Excel/Python 분석 (1시간)
  4. 그래프 생성 (30분)
  5. PPT 작성 (1시간)
  6. 이메일 발송 (10분)
- 총 소요: 3시간 15분

After (Agent):
- 자연어 요청만으로 자동 처리
- 소요 시간: 8분

월 4회 리포트 시:
- 시간 절감: 13시간 → 32분
- 인건비 절감: $800/월
- 더 자주 분석 가능 (월 4회 → 매일)
"""
```

---

## 실전 유즈케이스 3: 콘텐츠 리서치 & 생성 (Content Research & Generation)

```python
"""
시나리오: "AI 트렌드에 대한 블로그 포스트를 작성해줘 - 최신 정보 리서치, 1500-2000 단어, SEO 최적화, 이미지 포함, WordPress에 draft로 발행"

필요한 MCP:
1. Web Search MCP - 최신 정보 수집
2. Web Scrape MCP - 관련 기사 내용 추출
3. Database MCP - 과거 포스트 검색 (중복 방지)
4. File System MCP - 이미지 저장
5. CMS MCP - WordPress/Notion 발행
"""

# ============= Web Search MCP =============
class WebSearchMCP(Server):
    """웹 검색 MCP 서버 (Brave Search API 사용)"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="search_web",
                description="웹 검색. 매개변수: query, count (결과 개수), freshness (검색 기간)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색 쿼리"
                        },
                        "count": {
                            "type": "integer",
                            "description": "결과 개수 (기본 10)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 20
                        },
                        "freshness": {
                            "type": "string",
                            "description": "검색 기간",
                            "enum": ["day", "week", "month", "year"],
                            "default": "month"
                        }
                    },
                    "required": ["query"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "search_web":
            import httpx

            query = arguments['query']
            count = arguments.get('count', 10)
            freshness = arguments.get('freshness', 'month')

            # Brave Search API 호출
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={
                        "q": query,
                        "count": count,
                        "freshness": freshness
                    },
                    headers={"X-Subscription-Token": "YOUR_API_KEY"}
                )

            results = response.json()

            # 검색 결과 포맷
            formatted_results = []
            for i, item in enumerate(results.get('web', {}).get('results', []), 1):
                formatted_results.append(
                    f"{i}. **{item['title']}**\n"
                    f"   URL: {item['url']}\n"
                    f"   {item['description']}\n"
                )

            return [TextContent(
                type="text",
                text=f"검색 결과 ({len(formatted_results)}건):\n\n" + "\n".join(formatted_results)
            )]


# ============= Web Scrape MCP =============
class WebScrapeMCP(Server):
    """웹 스크래핑 MCP 서버"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="scrape_url",
                description="URL 스크래핑. 매개변수: url, selector (CSS selector, 선택)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "스크래핑할 URL"},
                        "selector": {
                            "type": "string",
                            "description": "CSS selector (선택, 지정하지 않으면 전체 텍스트)"
                        },
                        "extract_links": {
                            "type": "boolean",
                            "description": "링크 추출 여부",
                            "default": False
                        }
                    },
                    "required": ["url"]
                }
            ),
            Tool(
                name="extract_article",
                description="기사 본문 추출 (자동). 매개변수: url",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"}
                    },
                    "required": ["url"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "extract_article":
            import httpx
            from bs4 import BeautifulSoup
            from readability import Document

            url = arguments['url']

            # URL 다운로드
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                html = response.text

            # Readability로 본문 추출
            doc = Document(html)
            title = doc.title()
            content = doc.summary()

            # BeautifulSoup로 텍스트만 추출
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)

            return [TextContent(
                type="text",
                text=f"제목: {title}\n\nURL: {url}\n\n본문:\n{text[:2000]}..."
            )]


# ============= CMS MCP =============
class CMSMCP(Server):
    """CMS 발행 MCP 서버 (WordPress)"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="publish_wordpress",
                description="WordPress 포스트 발행. 매개변수: title, content, tags, featured_image, status",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string", "description": "HTML 또는 Markdown"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "category": {"type": "string"},
                        "featured_image": {"type": "string", "description": "이미지 URL"},
                        "status": {
                            "type": "string",
                            "enum": ["draft", "publish", "private"],
                            "default": "draft"
                        }
                    },
                    "required": ["title", "content"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "publish_wordpress":
            import httpx

            # WordPress REST API로 포스트 생성
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://yourblog.com/wp-json/wp/v2/posts",
                    json={
                        "title": arguments['title'],
                        "content": arguments['content'],
                        "status": arguments.get('status', 'draft'),
                        "tags": arguments.get('tags', []),
                        "categories": [arguments.get('category')],
                    },
                    auth=("admin", "password")
                )

            post_data = response.json()

            return [TextContent(
                type="text",
                text=f"WordPress 포스트 발행 완료\n"
                     f"제목: {post_data['title']['rendered']}\n"
                     f"URL: {post_data['link']}\n"
                     f"상태: {post_data['status']}\n"
                     f"ID: {post_data['id']}"
            )]


# ============= Agent 실행 =============
user_request = "AI 에이전트 트렌드에 대한 블로그 포스트를 작성해줘. 최신 정보 리서치, 1500-2000 단어, SEO 최적화, 이미지 포함, WordPress에 draft로 발행"

"""
Agent 자동 실행 순서:

STEP 1: search_web (Web Search MCP)
Agent 생각: "먼저 AI 에이전트 트렌드에 대한 최신 정보를 검색하자"

Query: "AI agent trends 2024"
Count: 10
Freshness: "month"

결과:
1. **The Rise of AI Agents in 2024: What You Need to Know**
   URL: https://techcrunch.com/ai-agents-2024
   AI agents are transforming how businesses automate tasks...

2. **Multi-Agent Systems: The Future of AI**
   URL: https://arxiv.org/ai-multiagent-systems
   Research shows multi-agent collaboration improves performance by 45%...

3. **Model Context Protocol (MCP): Standardizing AI Agent Tools**
   URL: https://anthropic.com/mcp-announcement
   Anthropic announces MCP for unified AI agent tooling...

[...7개 더]


STEP 2: extract_article (Web Scrape MCP) x 5번
Agent 생각: "상위 5개 기사의 본문을 추출해서 핵심 내용을 파악하자"

Article 1: TechCrunch 기사
제목: The Rise of AI Agents in 2024
핵심 내용:
- AI 에이전트 시장 2024년 $15B 예상
- Multi-agent systems 45% 성능 향상
- Enterprise 도입 증가 (전년 대비 230%)
- 주요 플레이어: OpenAI, Anthropic, Google

Article 2: arXiv 논문
제목: Multi-Agent Systems Research
핵심 내용:
- Agent 간 협업으로 복잡한 문제 해결
- Communication protocols 중요성
- 실제 적용 사례: 코드 생성, 데이터 분석

Article 3: Anthropic 블로그
제목: MCP Announcement
핵심 내용:
- Model Context Protocol (MCP) 표준화
- AI 에이전트가 도구/데이터에 접근하는 표준
- 마이크로서비스처럼 조합 가능

[Article 4, 5...]


STEP 3: query_database (Database MCP)
Agent 생각: "과거에 AI 에이전트 관련 포스트가 있었는지 확인해서 중복을 피하자"

SQL:
```sql
SELECT title, published_date, url
FROM blog_posts
WHERE (title LIKE '%AI agent%' OR tags LIKE '%AI agent%')
  AND published_date > DATE_SUB(NOW(), INTERVAL 6 MONTH)
ORDER BY published_date DESC
LIMIT 5
```

결과:
| title                                    | published_date | url                          |
|------------------------------------------|----------------|------------------------------|
| Introduction to AI Chatbots              | 2023-11-15     | /blog/ai-chatbots-intro      |
| AutoGPT: Autonomous AI Explained         | 2023-10-20     | /blog/autogpt-explained      |

Agent 판단: "AI 에이전트 트렌드는 없었네. 새로운 각도로 작성 가능!"


STEP 4: search_web (Web Search MCP)
Agent 생각: "관련 이미지를 찾자"

Query: "AI agent architecture diagram 2024"
Count: 5

결과:
1. https://example.com/images/ai-agent-architecture.png
2. https://another.com/multi-agent-system.jpg
...


STEP 5: LLM 콘텐츠 생성
Agent가 수집한 정보를 기반으로 블로그 포스트 작성:

```markdown
# 2024년 AI 에이전트 트렌드: 기업이 주목해야 할 5가지

## 서론

AI 에이전트가 단순한 챗봇을 넘어 복잡한 작업을 자율적으로 수행하는 시대가 왔습니다.
2024년 AI 에이전트 시장은 $15B 규모로 성장할 것으로 예상되며, 기업들의 도입이
전년 대비 230% 증가하고 있습니다.

이 글에서는 AI 에이전트의 최신 트렌드와 기업이 주목해야 할 5가지 핵심 요소를
살펴보겠습니다.

## 1. Multi-Agent Systems: 협업의 시대

### 개념
Multi-agent systems는 여러 AI 에이전트가 협업해서 복잡한 문제를 해결하는 방식입니다.

### 성능 향상
최근 arXiv 논문에 따르면, multi-agent 협업은 단일 에이전트 대비 **45% 성능 향상**을
보였습니다.

### 실제 사례
- **코드 생성**: 한 에이전트는 설계, 다른 에이전트는 구현, 또 다른 에이전트는 테스트
- **데이터 분석**: 데이터 수집 에이전트 + 분석 에이전트 + 시각화 에이전트
- **고객 지원**: 질문 분류 에이전트 + 답변 에이전트 + 에스컬레이션 에이전트

## 2. MCP (Model Context Protocol): 표준화의 시작

### 무엇인가?
Anthropic이 발표한 MCP는 AI 에이전트가 도구와 데이터에 접근하는 **표준 프로토콜**입니다.

### 왜 중요한가?
마치 HTTP가 웹을 표준화한 것처럼, MCP는 AI 에이전트 생태계를 표준화합니다.

### 마이크로서비스와의 유사성
각 MCP 서버는 특정 기능을 제공하고 (Database, Email, File System),
AI 에이전트가 이들을 조합해서 복잡한 작업을 수행합니다.

```python
# MCP 서버 예시
Database MCP → 데이터 조회
Email MCP → 이메일 발송
CRM MCP → 고객 interaction 기록

# Agent가 조합
"고객 john@example.com의 주문 상태 확인 후 이메일 발송"
→ Database MCP 호출
→ Email MCP 호출
→ CRM MCP 호출
```

## 3. Enterprise 도입 급증

### 통계
- 2023: Fortune 500 기업의 12%가 AI 에이전트 도입
- 2024: 39%로 증가 (230% 성장)

### 도입 분야
1. **고객 지원 자동화** (45%)
2. **데이터 분석** (32%)
3. **코드 리뷰** (18%)
4. **콘텐츠 생성** (15%)

### ROI
평균 ROI: 350%
- 인건비 절감: 40-60%
- 응답 속도: 10배 향상
- 정확도: 95% 이상

## 4. On-Premise vs Cloud: 비용 최적화

### 클라우드 API의 문제
- ChatGPT API: $10,000-50,000/월
- 높은 변동 비용
- 데이터 프라이버시 우려

### On-Premise AI 에이전트
- 초기 투자: $30,000
- 운영 비용: $500-2,000/월
- 90-95% 비용 절감
- 데이터 통제권 확보

### 하이브리드 접근
```
Rule-based (40%) → 즉시 응답, $0
Classical ML (30%) → 50ms, 최소 비용
RAG (20%) → 50ms, 약간의 비용
LLM (10%) → 500ms, 높은 비용

결과: 전체 비용 90% 감소
```

## 5. 실무 구현 패턴

### 패턴 1: Sequential (순차 실행)
```
사용자 요청
→ Agent가 Task 1 실행
→ 결과 확인
→ Task 2 실행
→ ...
```

### 패턴 2: Parallel (병렬 실행)
```
사용자 요청
→ Agent가 동시에 여러 MCP 호출
→ 결과 종합
```

### 패턴 3: Hierarchical (계층적)
```
Master Agent
  → Planning Agent (계획 수립)
  → Execution Agent (실행)
  → Review Agent (검토)
```

## 결론

AI 에이전트는 2024년 가장 주목받는 기술입니다. Multi-agent systems, MCP 표준화,
Enterprise 도입 급증, 비용 최적화, 다양한 구현 패턴을 이해하고 적용하는 기업이
경쟁 우위를 확보할 것입니다.

지금 바로 AI 에이전트 도입을 검토해보세요!

---

*이 글이 도움이 되었다면 공유해주세요!*

**태그**: #AI #Agent #Automation #MCP #EnterpriseAI
```

단어 수: 1,847 단어 ✓


STEP 6: SEO 최적화
Agent 생각: "SEO 최적화를 위해 메타 데이터 추가하자"

- **Title**: 2024년 AI 에이전트 트렌드: 기업이 주목해야 할 5가지
- **Meta Description**: AI 에이전트 트렌드 2024 - Multi-agent systems, MCP 표준화, Enterprise 도입 급증, 비용 최적화, 구현 패턴을 상세히 알아보세요.
- **Keywords**: AI 에이전트, Multi-agent, MCP, Model Context Protocol, 기업 AI, AI 자동화
- **Slug**: ai-agent-trends-2024
- **Featured Image**: https://example.com/images/ai-agent-architecture.png


STEP 7: publish_wordpress (CMS MCP)
Agent가 WordPress에 발행:

Request:
```json
{
  "title": "2024년 AI 에이전트 트렌드: 기업이 주목해야 할 5가지",
  "content": "[생성된 1,847단어 Markdown을 HTML로 변환]",
  "tags": ["AI", "Agent", "Automation", "MCP", "Enterprise"],
  "category": "Technology Trends",
  "featured_image": "https://example.com/images/ai-agent-architecture.png",
  "status": "draft",
  "meta": {
    "description": "AI 에이전트 트렌드 2024 - Multi-agent systems, MCP 표준화...",
    "keywords": "AI 에이전트, Multi-agent, MCP, Model Context Protocol"
  }
}
```

결과:
```
WordPress 포스트 발행 완료
제목: 2024년 AI 에이전트 트렌드: 기업이 주목해야 할 5가지
URL: https://yourblog.com/ai-agent-trends-2024/
상태: draft
ID: 1234
```


STEP 8: 최종 응답
"AI 에이전트 트렌드 블로그 포스트가 작성되었습니다!

📊 콘텐츠 통계:
- 단어 수: 1,847 단어
- 검색 기사: 10개 분석
- 주요 섹션: 5개
- 코드 예시: 3개 포함
- SEO 점수: 92/100

📝 주요 내용:
1. Multi-Agent Systems (협업의 시대)
2. MCP 표준화
3. Enterprise 도입 급증 (230% 성장)
4. On-Premise vs Cloud 비용 비교
5. 실무 구현 패턴 3가지

🔖 SEO 최적화:
- Title, Meta Description 설정
- 키워드: AI 에이전트, Multi-agent, MCP
- Featured Image 포함
- Internal linking 권장

📍 WordPress:
- 상태: Draft (검토 필요)
- URL: https://yourblog.com/ai-agent-trends-2024/
- 카테고리: Technology Trends

검토 후 발행하시면 됩니다!"

처리 시간: 3분
"""

"""
ROI:

Before (수동):
- 콘텐츠 작가가 수동 작업
  1. 트렌드 리서치 (2-3시간)
  2. 기사 읽기 (1시간)
  3. 초안 작성 (3-4시간)
  4. 이미지 찾기 (30분)
  5. SEO 최적화 (30분)
  6. 편집 (1시간)
- 총 소요: 8-10시간

After (Agent):
- 리서치 + 초안 생성: 3분
- 사람은 편집/검토만 (30분)
- 총 소요: 33분

주간 3개 포스트 작성 시:
- 시간 절감: 30시간 → 1.5시간 (95% 절감)
- 인건비 절감: $1,500/주
- 더 자주 발행 가능 (주 3개 → 매일)
- SEO 일관성 향상 (자동화된 키워드 분석)
"""
```

---

## 실전 유즈케이스 4: 코드 리뷰 자동화 (Automated Code Review)

```python
"""
시나리오: "mycompany/backend PR #123 코드 리뷰 해줘"

필요한 MCP:
1. GitHub MCP - PR 정보, diff 조회, 코멘트 작성
2. File System MCP - 로컬 코드 읽기
3. Static Analysis MCP - ESLint, Pylint 실행
4. Security MCP - 보안 취약점 스캔 (Snyk, Bandit)
5. Test MCP - 테스트 실행, 커버리지 확인
"""

# ============= GitHub MCP =============
class GitHubMCP(Server):
    """GitHub 연동 MCP 서버"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="get_pr_info",
                description="PR 기본 정보 조회. 매개변수: repo, pr_number",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "owner/repo"},
                        "pr_number": {"type": "integer"}
                    },
                    "required": ["repo", "pr_number"]
                }
            ),
            Tool(
                name="get_pr_diff",
                description="PR diff 조회. 매개변수: repo, pr_number",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string"},
                        "pr_number": {"type": "integer"}
                    },
                    "required": ["repo", "pr_number"]
                }
            ),
            Tool(
                name="get_pr_files",
                description="PR에서 변경된 파일 목록. 매개변수: repo, pr_number",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string"},
                        "pr_number": {"type": "integer"}
                    },
                    "required": ["repo", "pr_number"]
                }
            ),
            Tool(
                name="create_review_comment",
                description="리뷰 코멘트 작성. 매개변수: repo, pr_number, body, path, line",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string"},
                        "pr_number": {"type": "integer"},
                        "body": {"type": "string"},
                        "path": {"type": "string", "description": "파일 경로"},
                        "line": {"type": "integer"}
                    },
                    "required": ["repo", "pr_number", "body", "path"]
                }
            ),
            Tool(
                name="submit_review",
                description="PR 리뷰 제출. 매개변수: repo, pr_number, event, body",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string"},
                        "pr_number": {"type": "integer"},
                        "event": {
                            "type": "string",
                            "enum": ["APPROVE", "REQUEST_CHANGES", "COMMENT"]
                        },
                        "body": {"type": "string"}
                    },
                    "required": ["repo", "pr_number", "event"]
                }
            )
        ]


# ============= Static Analysis MCP =============
class StaticAnalysisMCP(Server):
    """정적 분석 MCP 서버"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="run_pylint",
                description="Pylint 실행 (Python). 매개변수: files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "rcfile": {"type": "string", "description": ".pylintrc 경로 (선택)"}
                    },
                    "required": ["files"]
                }
            ),
            Tool(
                name="check_complexity",
                description="코드 복잡도 체크 (Cyclomatic Complexity). 매개변수: file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "max_complexity": {"type": "integer", "default": 10}
                    },
                    "required": ["file"]
                }
            ),
            Tool(
                name="check_code_style",
                description="코드 스타일 체크 (Black, Flake8). 매개변수: files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string"}},
                        "tool": {
                            "type": "string",
                            "enum": ["black", "flake8", "autopep8"]
                        }
                    },
                    "required": ["files"]
                }
            )
        ]


# ============= Security MCP =============
class SecurityMCP(Server):
    """보안 스캔 MCP 서버"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="scan_dependencies",
                description="의존성 보안 스캔 (Snyk). 매개변수: directory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string"},
                        "severity_threshold": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"],
                            "default": "medium"
                        }
                    },
                    "required": ["directory"]
                }
            ),
            Tool(
                name="scan_secrets",
                description="하드코딩된 시크릿 검사 (Trufflehog). 매개변수: files",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["files"]
                }
            ),
            Tool(
                name="check_sql_injection",
                description="SQL 인젝션 취약점 검사 (Bandit). 매개변수: file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"}
                    },
                    "required": ["file"]
                }
            ),
            Tool(
                name="scan_xss",
                description="XSS 취약점 검사. 매개변수: file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"}
                    },
                    "required": ["file"]
                }
            )
        ]


# ============= Test MCP =============
class TestMCP(Server):
    """테스트 실행 MCP 서버"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="run_tests",
                description="테스트 실행 (pytest, unittest). 매개변수: directory, pattern",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string", "description": "테스트 디렉토리"},
                        "pattern": {"type": "string", "description": "파일 패턴 (예: test_*.py)"},
                        "coverage": {"type": "boolean", "description": "커버리지 측정 여부", "default": True}
                    },
                    "required": ["directory"]
                }
            ),
            Tool(
                name="get_coverage_report",
                description="테스트 커버리지 리포트 조회. 매개변수: format",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["summary", "detailed", "html"],
                            "default": "summary"
                        }
                    }
                }
            )
        ]


# ============= Agent 실행 =============
user_request = "mycompany/backend PR #123 코드 리뷰 해줘"

"""
Agent 자동 실행 순서:

STEP 1: get_pr_info (GitHub MCP)
Agent 생각: "PR 기본 정보부터 확인하자"

Request: {"repo": "mycompany/backend", "pr_number": 123}

결과:
```json
{
  "number": 123,
  "title": "Add user authentication endpoint",
  "author": "john-doe",
  "state": "open",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T14:20:00Z",
  "additions": 324,
  "deletions": 12,
  "changed_files": 8,
  "mergeable": true
}
```


STEP 2: get_pr_files (GitHub MCP)
Agent 생각: "어떤 파일들이 변경되었는지 확인하자"

결과:
```json
[
  {
    "filename": "src/auth/login.py",
    "status": "modified",
    "additions": 120,
    "deletions": 5,
    "changes": 125
  },
  {
    "filename": "src/auth/middleware.py",
    "status": "added",
    "additions": 80,
    "deletions": 0,
    "changes": 80
  },
  {
    "filename": "src/models/user.py",
    "status": "modified",
    "additions": 45,
    "deletions": 3,
    "changes": 48
  },
  {
    "filename": "tests/test_auth.py",
    "status": "modified",
    "additions": 79,
    "deletions": 4,
    "changes": 83
  },
  {
    "filename": "requirements.txt",
    "status": "modified",
    "additions": 1,
    "deletions": 0,
    "changes": 1
  }
]
```


STEP 3: get_pr_diff (GitHub MCP)
Agent 생각: "변경된 코드를 자세히 보자"

결과 (일부):
```diff
diff --git a/src/auth/login.py b/src/auth/login.py
index 1234567..abcdefg 100644
--- a/src/auth/login.py
+++ b/src/auth/login.py
@@ -10,6 +10,15 @@ from flask import Blueprint, request, jsonify
+def authenticate_user(email, password):
+    # TODO: Implement rate limiting
+    query = f"SELECT * FROM users WHERE email = '{email}'"
+    user = db.execute(query).fetchone()
+
+    if user and user.password == password:
+        return generate_token(user.id)
+    return None

diff --git a/src/auth/middleware.py b/src/auth/middleware.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/src/auth/middleware.py
@@ -0,0 +1,80 @@
+JWT_SECRET = "mysecretkey123"  # TODO: Move to env
+
+def verify_token(token):
+    try:
+        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
+        return payload
+    except:
+        return None
```


STEP 4: run_pylint (Static Analysis MCP)
Agent 생각: "Python 코드 품질 체크하자"

Request: {
  "files": ["src/auth/login.py", "src/auth/middleware.py", "src/models/user.py"]
}

결과:
```
Pylint 분석 완료

총 5개 이슈:

1. src/auth/login.py:13
   [W0612] Unused import 'datetime'

2. src/auth/middleware.py:23
   [C0103] Variable name 'JWT_SECRET' doesn't conform to snake_case

3. src/auth/middleware.py:45
   [R1702] Too many nested blocks (6/5)
   Complexity: 15 (권장: 10 이하)

4. src/models/user.py:78
   [C0103] Argument name 'Id' doesn't conform to snake_case

5. src/auth/login.py:45
   [W0703] Broad except clause
```


STEP 5: check_complexity (Static Analysis MCP)
Agent 생각: "복잡도가 높은 함수를 확인하자"

Request: {"file": "src/auth/middleware.py", "max_complexity": 10}

결과:
```
복잡도 체크 완료

함수별 Cyclomatic Complexity:

❌ verify_and_refresh_token() - 복잡도 15 (권장: 10 이하)
   위치: src/auth/middleware.py:45
   권장: 여러 함수로 분리

✅ verify_token() - 복잡도 3
✅ generate_token() - 복잡도 2
```


STEP 6: scan_dependencies (Security MCP)
Agent 생각: "의존성 보안 취약점을 확인하자"

Request: {"directory": ".", "severity_threshold": "medium"}

결과:
```
의존성 보안 스캔 완료 (Snyk)

발견된 취약점:

🚨 CRITICAL (1개):
- flask==2.0.1
  CVE-2023-30861: Werkzeug cookie parsing vulnerability
  권장: flask>=2.3.0 업그레이드

⚠️ HIGH (2개):
- requests==2.25.1
  CVE-2023-32681: Proxy-Authorization header leak
  권장: requests>=2.31.0 업그레이드

- cryptography==38.0.1
  CVE-2023-38325: Cipher vulnerability
  권장: cryptography>=41.0.0 업그레이드

ℹ️ MEDIUM (3개):
...
```


STEP 7: scan_secrets (Security MCP)
Agent 생각: "하드코딩된 시크릿이 있는지 확인하자"

Request: {
  "files": ["src/auth/login.py", "src/auth/middleware.py", "src/models/user.py"]
}

결과:
```
시크릿 스캔 완료 (Trufflehog)

발견된 하드코딩 시크릿:

🚨 HIGH: JWT Secret Key
   파일: src/auth/middleware.py:1
   코드: JWT_SECRET = "mysecretkey123"
   권장: 환경 변수로 이동

🚨 MEDIUM: API Key 패턴
   파일: src/auth/login.py:67
   코드: api_key = "sk-..."
   권장: 환경 변수로 이동
```


STEP 8: check_sql_injection (Security MCP)
Agent 생각: "SQL 인젝션 취약점을 확인하자"

Request: {"file": "src/auth/login.py"}

결과:
```
SQL 인젝션 스캔 완료 (Bandit)

🚨 CRITICAL: SQL Injection 취약점 발견!

파일: src/auth/login.py:13
코드:
```python
query = f"SELECT * FROM users WHERE email = '{email}'"
user = db.execute(query).fetchone()
```

위험도: CRITICAL
설명: f-string으로 SQL 쿼리를 생성하면 SQL 인젝션 공격에 취약합니다.

공격 예시:
```python
email = "admin' OR '1'='1"
# 결과 쿼리: SELECT * FROM users WHERE email = 'admin' OR '1'='1'
# → 모든 사용자 정보 노출!
```

권장 수정:
```python
query = "SELECT * FROM users WHERE email = ?"
user = db.execute(query, (email,)).fetchone()
```
```


STEP 9: run_tests (Test MCP)
Agent 생각: "테스트를 실행해서 통과하는지 확인하자"

Request: {"directory": "tests", "pattern": "test_*.py", "coverage": True}

결과:
```
테스트 실행 완료 (pytest + coverage)

=== 테스트 결과 ===
✅ Passed: 42
❌ Failed: 1
⏭ Skipped: 2

실패한 테스트:
tests/test_auth.py::test_invalid_token_rejection
  AssertionError: Expected 401, got 500

=== 커버리지 ===
전체: 79% → 85% (+6%)

파일별 커버리지:
src/auth/login.py: 92%
src/auth/middleware.py: 78%
src/models/user.py: 95%

Missing 커버리지:
src/auth/middleware.py:67-72 (에러 핸들링)
```


STEP 10: LLM 종합 분석
Agent가 모든 정보를 종합해서 리뷰 작성:

분석 결과:
- 정적 분석: 5개 이슈
- 복잡도: 1개 함수 높음
- 보안 취약점: CRITICAL 2개, HIGH 2개
- 테스트: 1개 실패, 커버리지 양호

판단: REQUEST_CHANGES (중대한 보안 이슈)


STEP 11: create_review_comment (GitHub MCP) x 7번
Agent가 각 이슈에 대해 코멘트 작성:

Comment 1 (src/auth/login.py:13):
```markdown
🚨 **CRITICAL: SQL Injection 취약점**

현재 코드는 SQL 인젝션 공격에 매우 취약합니다.

**문제 코드:**
```python
query = f"SELECT * FROM users WHERE email = '{email}'"
user = db.execute(query).fetchone()
```

**공격 시나리오:**
```python
# 공격자가 이메일란에 입력:
email = "admin' OR '1'='1"

# 실제 실행되는 쿼리:
SELECT * FROM users WHERE email = 'admin' OR '1'='1'
# → 모든 사용자 정보 노출!
```

**권장 수정:**
```python
query = "SELECT * FROM users WHERE email = ?"
user = db.execute(query, (email,)).fetchone()

# 또는 ORM 사용:
user = User.query.filter_by(email=email).first()
```

**참고**: OWASP Top 10 #1 취약점입니다. 즉시 수정이 필요합니다.
```

Comment 2 (src/auth/middleware.py:1):
```markdown
⚠️ **보안 이슈: JWT 시크릿 하드코딩**

JWT 시크릿이 코드에 하드코딩되어 있습니다.

**문제:**
```python
JWT_SECRET = "mysecretkey123"
```

**위험성:**
- Git 히스토리에 시크릿 노출
- 코드 유출 시 모든 토큰 위조 가능
- 시크릿 변경 시 코드 재배포 필요

**권장 수정:**
```python
import os

JWT_SECRET = os.getenv('JWT_SECRET')
if not JWT_SECRET:
    raise ValueError("JWT_SECRET environment variable is required")
```

**추가 작업:**
1. `.env.example` 파일 생성
2. `.gitignore`에 `.env` 추가
3. 배포 환경에 환경 변수 설정
```

Comment 3 (src/auth/middleware.py:45):
```markdown
📊 **코드 복잡도 높음**

`verify_and_refresh_token()` 함수의 Cyclomatic Complexity가 15입니다 (권장: 10 이하).

**문제:**
- 유지보수 어려움
- 테스트 어려움
- 버그 발생 가능성 높음

**권장:**
여러 함수로 분리하세요.

```python
# Before: 하나의 큰 함수 (복잡도 15)
def verify_and_refresh_token(token):
    # 80줄의 복잡한 로직...

# After: 여러 작은 함수로 분리
def verify_token(token):
    # 토큰 검증만

def check_expiration(payload):
    # 만료 확인만

def refresh_if_needed(payload):
    # 갱신 로직만

def verify_and_refresh_token(token):
    payload = verify_token(token)
    if check_expiration(payload):
        return refresh_if_needed(payload)
    return payload
```
```

Comment 4 (requirements.txt):
```markdown
🚨 **의존성 보안 취약점 (CRITICAL)**

여러 패키지에서 중대한 보안 취약점이 발견되었습니다.

| 패키지 | 현재 버전 | 권장 버전 | 심각도 | CVE |
|--------|-----------|-----------|--------|-----|
| flask | 2.0.1 | ≥2.3.0 | CRITICAL | CVE-2023-30861 |
| requests | 2.25.1 | ≥2.31.0 | HIGH | CVE-2023-32681 |
| cryptography | 38.0.1 | ≥41.0.0 | HIGH | CVE-2023-38325 |

**즉시 업그레이드가 필요합니다:**
```diff
- flask==2.0.1
+ flask==2.3.2
- requests==2.25.1
+ requests==2.31.0
- cryptography==38.0.1
+ cryptography==41.0.4
```

**업그레이드 후 확인 사항:**
1. 테스트 실행
2. Breaking changes 확인
3. 성능 테스트
```

Comment 5 (tests/test_auth.py):
```markdown
❌ **테스트 실패**

`test_invalid_token_rejection` 테스트가 실패했습니다.

**실패 내용:**
```
AssertionError: Expected 401 Unauthorized, got 500 Internal Server Error
```

**문제 원인 추정:**
`src/auth/middleware.py:45`의 예외 처리가 없어서 500 에러 발생

**권장 수정:**
```python
def verify_token(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload, None
    except jwt.ExpiredSignatureError:
        return None, "Token expired"
    except jwt.InvalidTokenError:
        return None, "Invalid token"
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None, "Token verification failed"

# API에서:
payload, error = verify_token(token)
if error:
    return jsonify({"error": error}), 401  # 401로 명확히 반환
```
```

Comment 6 (src/auth/login.py:45):
```markdown
ℹ️ **사용하지 않는 import**

Line 3의 `datetime` import가 사용되지 않습니다.

```python
from datetime import datetime  # 사용하지 않음
```

**수정:**
삭제하거나 실제로 사용하세요.
```

Comment 7 (general):
```markdown
📈 **테스트 커버리지**

전체 커버리지가 79% → 85%로 향상되었습니다! 👍

**Missing 커버리지:**
- `src/auth/middleware.py:67-72` (에러 핸들링)

**권장:**
에러 핸들링에 대한 테스트 추가:
```python
def test_token_verification_network_error():
    # 네트워크 에러 시나리오
    with mock.patch('jwt.decode', side_effect=NetworkError):
        result = verify_token("valid_token")
        assert result is None
```
```


STEP 12: submit_review (GitHub MCP)
Agent가 종합 리뷰 제출:

Request:
```json
{
  "repo": "mycompany/backend",
  "pr_number": 123,
  "event": "REQUEST_CHANGES",
  "body": "코드 리뷰 완료했습니다.

## 요약

전반적으로 인증 기능 구현은 잘 되었지만, **2건의 중대한 보안 이슈**가 발견되어 변경을 요청합니다.

### ✅ 좋은 점
- 테스트 커버리지 향상 (79% → 85%)
- 전반적인 코드 구조 깔끔
- 새로운 기능 추가로 인한 가치 향상

### 🚨 즉시 수정 필요 (CRITICAL)
1. **SQL Injection 취약점** (`src/auth/login.py:13`)
   - f-string으로 쿼리 생성 → 파라미터화된 쿼리 사용

2. **JWT 시크릿 하드코딩** (`src/auth/middleware.py:1`)
   - 코드에 시크릿 노출 → 환경 변수로 이동

3. **의존성 보안 취약점** (`requirements.txt`)
   - flask, requests, cryptography 업그레이드 필요

### ⚠️ 개선 권장
- 코드 복잡도 감소 (`verify_and_refresh_token` 함수 분리)
- 테스트 실패 수정 (`test_invalid_token_rejection`)
- 에러 핸들링 개선

### 📊 통계
- 파일 변경: 8개
- 추가 라인: +324
- 삭제 라인: -12
- 이슈 발견: 7개 (CRITICAL 2, HIGH 2, MEDIUM 3)
- 테스트: 42 passed, 1 failed
- 커버리지: 85%

**보안 이슈들을 수정한 후 재검토 부탁드립니다.** 수정 후 다시 리뷰 요청하시면 빠르게 확인하겠습니다!"
}
```


STEP 13: 최종 응답
"PR #123 코드 리뷰가 완료되었습니다.

🔍 리뷰 요약:
- 총 7개 코멘트 작성
- 상태: REQUEST_CHANGES

🚨 중대한 이슈 (즉시 수정 필요):
1. SQL Injection 취약점 (CRITICAL)
2. JWT 시크릿 하드코딩 (HIGH)
3. 의존성 보안 취약점 (CRITICAL)

⚠️ 개선 권장:
- 코드 복잡도 감소
- 테스트 1개 실패 수정
- 사용하지 않는 import 제거

✅ 긍정적인 부분:
- 테스트 커버리지 +6% 향상
- 깔끔한 코드 구조

GitHub에서 확인하세요:
https://github.com/mycompany/backend/pull/123

처리 시간: 45초"
"""

"""
ROI:

Before (수동):
- 시니어 개발자가 직접 리뷰
  1. PR 확인 (5분)
  2. 코드 읽기 (20분)
  3. 로컬에서 테스트 (10분)
  4. 보안 이슈 확인 (10분)
  5. 코멘트 작성 (15분)
- 총 소요: 60분
- 놓칠 수 있는 이슈 (특히 보안 취약점)
- 일관성 부족

After (Agent):
- 자동화된 첫 번째 리뷰: 45초
- 모든 보안/정적 분석 이슈 자동 감지
- 100% 일관성
- 사람은 로직/디자인만 집중

PR 하루 20개 처리 시:
- 시간 절감: 20시간 → 30분 (98% 절감)
- 보안 이슈 감지율: +80%
- 개발자는 창의적 작업에 집중
- 코드 품질 향상: 평균 +35%
"""
```

---

## 5. 금융/은행 (Banking & Finance): 실시간 이상거래 탐지 및 대응

### 시나리오
**요청**: "최근 1시간 동안 의심스러운 거래 패턴을 감지하고, 고위험 건은 자동으로 차단하고 담당자에게 리포트 보내줘"

### 필요한 MCP 서버들

```python
# 1. Transaction Database MCP
class TransactionDatabaseMCP(Server):
    """거래 데이터베이스 접근"""

    def __init__(self):
        super().__init__("transaction-db-mcp")
        self.pool = asyncpg.create_pool(
            "postgresql://bank_user:pass@db:5432/transactions"
        )

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="query_transactions",
                description="거래 내역 조회",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string"},
                        "time_range_hours": {"type": "number"},
                        "min_amount": {"type": "number"},
                        "transaction_types": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            ),
            Tool(
                name="get_account_info",
                description="계좌 정보 및 프로필 조회",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string"}
                    },
                    "required": ["account_id"]
                }
            ),
            Tool(
                name="get_transaction_history",
                description="과거 거래 패턴 분석을 위한 이력 조회",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string"},
                        "days": {"type": "number"}
                    },
                    "required": ["account_id"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "query_transactions":
            hours = arguments.get('time_range_hours', 1)

            query = """
                SELECT
                    t.transaction_id,
                    t.account_id,
                    t.amount,
                    t.currency,
                    t.transaction_type,
                    t.destination_account,
                    t.destination_country,
                    t.timestamp,
                    t.ip_address,
                    t.device_id,
                    a.customer_name,
                    a.account_type,
                    a.average_monthly_balance
                FROM transactions t
                JOIN accounts a ON t.account_id = a.account_id
                WHERE t.timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY t.timestamp DESC
            """

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, hours)

            transactions = [dict(row) for row in rows]

            return [TextContent(
                type="text",
                text=json.dumps(transactions, default=str, ensure_ascii=False)
            )]

        elif name == "get_account_info":
            account_id = arguments['account_id']

            query = """
                SELECT
                    account_id,
                    customer_name,
                    customer_ssn_hash,
                    phone,
                    email,
                    account_type,
                    account_status,
                    creation_date,
                    average_monthly_balance,
                    credit_score,
                    kyc_status,
                    risk_rating
                FROM accounts
                WHERE account_id = $1
            """

            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, account_id)

            return [TextContent(
                type="text",
                text=json.dumps(dict(row), default=str, ensure_ascii=False)
            )]

        elif name == "get_transaction_history":
            account_id = arguments['account_id']
            days = arguments['days']

            query = """
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as transaction_count,
                    SUM(amount) as total_amount,
                    AVG(amount) as avg_amount,
                    MAX(amount) as max_amount,
                    COUNT(DISTINCT destination_country) as unique_countries
                FROM transactions
                WHERE account_id = $1
                  AND timestamp > NOW() - INTERVAL '%s days'
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, account_id, days)

            history = [dict(row) for row in rows]

            return [TextContent(
                type="text",
                text=json.dumps(history, default=str, ensure_ascii=False)
            )]


# 2. Fraud Detection MCP
class FraudDetectionMCP(Server):
    """머신러닝 기반 사기 탐지 엔진"""

    def __init__(self):
        super().__init__("fraud-detection-mcp")
        self.model = self._load_model()

    def _load_model(self):
        """사전 훈련된 이상거래 탐지 모델 로드"""
        import joblib
        return joblib.load('/models/fraud_detection_v3.pkl')

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="analyze_transaction",
                description="단일 거래의 사기 확률 분석",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "transaction": {"type": "object"},
                        "account_history": {"type": "object"}
                    },
                    "required": ["transaction"]
                }
            ),
            Tool(
                name="batch_analyze",
                description="여러 거래를 동시에 분석",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "transactions": {
                            "type": "array",
                            "items": {"type": "object"}
                        }
                    },
                    "required": ["transactions"]
                }
            ),
            Tool(
                name="get_fraud_indicators",
                description="사기 지표 상세 분석",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "transaction_id": {"type": "string"}
                    },
                    "required": ["transaction_id"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "analyze_transaction":
            transaction = arguments['transaction']
            account_history = arguments.get('account_history', {})

            # Feature engineering
            features = self._extract_features(transaction, account_history)

            # ML 모델 예측
            fraud_probability = self.model.predict_proba([features])[0][1]

            # Risk level 결정
            if fraud_probability > 0.9:
                risk_level = "CRITICAL"
                action = "BLOCK_IMMEDIATELY"
            elif fraud_probability > 0.7:
                risk_level = "HIGH"
                action = "MANUAL_REVIEW"
            elif fraud_probability > 0.5:
                risk_level = "MEDIUM"
                action = "MONITOR"
            else:
                risk_level = "LOW"
                action = "ALLOW"

            # 사기 지표 분석
            indicators = self._analyze_indicators(transaction, account_history)

            result = {
                "transaction_id": transaction['transaction_id'],
                "fraud_probability": float(fraud_probability),
                "risk_level": risk_level,
                "recommended_action": action,
                "indicators": indicators,
                "confidence": 0.94
            }

            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False)
            )]

        elif name == "batch_analyze":
            transactions = arguments['transactions']
            results = []

            for txn in transactions:
                features = self._extract_features(txn, {})
                fraud_prob = self.model.predict_proba([features])[0][1]

                results.append({
                    "transaction_id": txn['transaction_id'],
                    "fraud_probability": float(fraud_prob),
                    "risk_level": "HIGH" if fraud_prob > 0.7 else "MEDIUM" if fraud_prob > 0.5 else "LOW"
                })

            return [TextContent(
                type="text",
                text=json.dumps(results, ensure_ascii=False)
            )]

    def _extract_features(self, transaction, account_history):
        """거래로부터 ML 특징 추출"""
        return [
            transaction['amount'],
            1 if transaction.get('destination_country') != 'KR' else 0,
            transaction.get('hour_of_day', 12),
            account_history.get('avg_amount', 100000),
            # ... 더 많은 특징들
        ]

    def _analyze_indicators(self, transaction, account_history):
        """사기 지표 분석"""
        indicators = []

        # 1. 비정상적인 금액
        avg_amount = account_history.get('avg_amount', 100000)
        if transaction['amount'] > avg_amount * 5:
            indicators.append({
                "type": "UNUSUAL_AMOUNT",
                "severity": "HIGH",
                "description": f"평균 거래액의 5배 초과 ({transaction['amount']:,}원)"
            })

        # 2. 해외 거래
        if transaction.get('destination_country') not in ['KR', 'US', 'JP']:
            indicators.append({
                "type": "HIGH_RISK_COUNTRY",
                "severity": "MEDIUM",
                "description": f"고위험 국가로의 송금: {transaction.get('destination_country')}"
            })

        # 3. 야간 거래
        hour = transaction.get('hour_of_day', 12)
        if hour < 6 or hour > 23:
            indicators.append({
                "type": "UNUSUAL_TIME",
                "severity": "MEDIUM",
                "description": f"비정상적인 거래 시간: {hour}시"
            })

        # 4. 짧은 시간 다중 거래
        if account_history.get('recent_transaction_count', 0) > 5:
            indicators.append({
                "type": "RAPID_TRANSACTIONS",
                "severity": "HIGH",
                "description": "1시간 내 5회 이상 거래"
            })

        return indicators


# 3. Transaction Control MCP
class TransactionControlMCP(Server):
    """거래 차단/승인 제어"""

    def __init__(self):
        super().__init__("transaction-control-mcp")
        self.redis_client = redis.Redis(host='redis', port=6379, db=0)

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="block_transaction",
                description="거래 즉시 차단",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "transaction_id": {"type": "string"},
                        "reason": {"type": "string"},
                        "block_duration_hours": {"type": "number"}
                    },
                    "required": ["transaction_id", "reason"]
                }
            ),
            Tool(
                name="flag_for_review",
                description="수동 검토 대기열에 추가",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "transaction_id": {"type": "string"},
                        "priority": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                        "notes": {"type": "string"}
                    },
                    "required": ["transaction_id"]
                }
            ),
            Tool(
                name="freeze_account",
                description="계좌 동결 (모든 거래 차단)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string"},
                        "reason": {"type": "string"},
                        "duration_hours": {"type": "number"}
                    },
                    "required": ["account_id", "reason"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "block_transaction":
            txn_id = arguments['transaction_id']
            reason = arguments['reason']

            # Redis에 차단 정보 저장
            self.redis_client.setex(
                f"blocked_txn:{txn_id}",
                arguments.get('block_duration_hours', 24) * 3600,
                json.dumps({"reason": reason, "timestamp": datetime.now().isoformat()})
            )

            # 실시간 차단 API 호출
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://payment-gateway:8000/api/v1/transactions/block",
                    json={"transaction_id": txn_id, "reason": reason}
                )

            return [TextContent(
                type="text",
                text=f"거래 {txn_id} 차단 완료: {reason}"
            )]

        elif name == "freeze_account":
            account_id = arguments['account_id']
            reason = arguments['reason']
            duration = arguments.get('duration_hours', 24)

            self.redis_client.setex(
                f"frozen_account:{account_id}",
                duration * 3600,
                json.dumps({"reason": reason, "timestamp": datetime.now().isoformat()})
            )

            return [TextContent(
                type="text",
                text=f"계좌 {account_id} 동결 완료 ({duration}시간)"
            )]


# 4. Alert & Reporting MCP
class BankingAlertMCP(Server):
    """알림 및 리포트 생성"""

    def __init__(self):
        super().__init__("banking-alert-mcp")

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="send_fraud_alert",
                description="사기 경보 발송 (이메일, SMS, Slack)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "recipients": {"type": "array", "items": {"type": "string"}},
                        "severity": {"type": "string"},
                        "transaction_details": {"type": "object"},
                        "analysis_result": {"type": "object"}
                    },
                    "required": ["recipients", "severity"]
                }
            ),
            Tool(
                name="generate_fraud_report",
                description="사기 분석 리포트 생성 (PDF)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "time_range_hours": {"type": "number"},
                        "flagged_transactions": {"type": "array"},
                        "include_charts": {"type": "boolean"}
                    },
                    "required": ["flagged_transactions"]
                }
            ),
            Tool(
                name="notify_customer",
                description="고객에게 알림 발송",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string"},
                        "message": {"type": "string"},
                        "channels": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["account_id", "message"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "send_fraud_alert":
            severity = arguments['severity']
            recipients = arguments['recipients']
            txn_details = arguments.get('transaction_details', {})
            analysis = arguments.get('analysis_result', {})

            # Slack 알림
            slack_message = {
                "text": f"🚨 {severity} 사기 거래 탐지",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{severity} 사기 거래 탐지*\n\n" +
                                    f"거래 ID: `{txn_details.get('transaction_id')}`\n" +
                                    f"금액: {txn_details.get('amount'):,}원\n" +
                                    f"사기 확률: {analysis.get('fraud_probability', 0)*100:.1f}%\n" +
                                    f"권장 조치: {analysis.get('recommended_action')}"
                        }
                    }
                ]
            }

            async with httpx.AsyncClient() as client:
                await client.post(
                    "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                    json=slack_message
                )

            # 이메일 발송
            email_body = f"""
            <h2>사기 거래 탐지 알림</h2>
            <p><strong>심각도:</strong> {severity}</p>
            <p><strong>거래 ID:</strong> {txn_details.get('transaction_id')}</p>
            <p><strong>금액:</strong> {txn_details.get('amount'):,}원</p>
            <p><strong>사기 확률:</strong> {analysis.get('fraud_probability', 0)*100:.1f}%</p>
            <p><strong>권장 조치:</strong> {analysis.get('recommended_action')}</p>
            """

            # 이메일 발송 로직 (생략)

            return [TextContent(
                type="text",
                text=f"알림 발송 완료: {len(recipients)}명에게 전송됨"
            )]

        elif name == "generate_fraud_report":
            flagged = arguments['flagged_transactions']

            # PDF 리포트 생성
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            pdf_path = f"/tmp/fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            c = canvas.Canvas(pdf_path, pagesize=letter)

            c.drawString(100, 750, "사기 거래 분석 리포트")
            c.drawString(100, 730, f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            c.drawString(100, 710, f"탐지된 의심 거래: {len(flagged)}건")

            y = 680
            for txn in flagged[:10]:  # 상위 10건
                c.drawString(120, y, f"• 거래 ID: {txn.get('transaction_id')}, " +
                                     f"금액: {txn.get('amount'):,}원, " +
                                     f"위험도: {txn.get('risk_level')}")
                y -= 20

            c.save()

            return [TextContent(
                type="text",
                text=f"리포트 생성 완료: {pdf_path}"
            )]
```

### Agent 실행 시퀀스

```
사용자: "최근 1시간 동안 의심스러운 거래 패턴을 감지하고, 고위험 건은 자동으로 차단하고 담당자에게 리포트 보내줘"

Agent 생각: "실시간 사기 탐지 작업이군. 다음 단계로 진행해야겠다:
1. 최근 1시간 모든 거래 조회
2. 각 거래에 대해 사기 확률 분석
3. 고위험 거래는 즉시 차단
4. 중위험 거래는 검토 대기열에 추가
5. 담당자에게 알림 및 리포트 발송"

STEP 1: 최근 거래 조회
→ transaction-db-mcp.query_transactions({
    "time_range_hours": 1,
    "min_amount": 0
})

Response: [
    {
        "transaction_id": "TXN-2025-001234",
        "account_id": "ACC-789012",
        "amount": 45000000,  # 4500만원
        "currency": "KRW",
        "transaction_type": "WIRE_TRANSFER",
        "destination_account": "ACC-999888",
        "destination_country": "CN",
        "timestamp": "2025-01-18 14:23:45",
        "ip_address": "203.142.67.89",
        "device_id": "unknown",
        "customer_name": "김철수",
        "account_type": "SAVINGS",
        "average_monthly_balance": 3000000
    },
    {
        "transaction_id": "TXN-2025-001235",
        "account_id": "ACC-123456",
        "amount": 12000000,  # 1200만원
        "currency": "KRW",
        "transaction_type": "ATM_WITHDRAWAL",
        "destination_country": "KR",
        "timestamp": "2025-01-18 14:45:12",
        ...
    },
    ... (총 248건 거래 조회됨)
]

Agent 생각: "248건의 거래가 있네. 이 중 고액 거래와 해외 송금을 우선적으로 분석해야겠다."


STEP 2: 일괄 사기 분석 (초기 스크리닝)
→ fraud-detection-mcp.batch_analyze({
    "transactions": [248건 전체]
})

Response: [
    {
        "transaction_id": "TXN-2025-001234",
        "fraud_probability": 0.94,
        "risk_level": "CRITICAL"
    },
    {
        "transaction_id": "TXN-2025-001235",
        "fraud_probability": 0.31,
        "risk_level": "LOW"
    },
    {
        "transaction_id": "TXN-2025-001240",
        "fraud_probability": 0.78,
        "risk_level": "HIGH"
    },
    {
        "transaction_id": "TXN-2025-001255",
        "fraud_probability": 0.88,
        "risk_level": "CRITICAL"
    },
    ... (CRITICAL 3건, HIGH 8건, MEDIUM 17건 탐지)
]

Agent 생각: "3건의 CRITICAL, 8건의 HIGH 위험 거래 발견! CRITICAL 건들은 즉시 차단하고, HIGH 건들은 상세 분석 후 결정해야겠다."


STEP 3: 첫 번째 CRITICAL 거래 상세 분석
→ transaction-db-mcp.get_account_info({
    "account_id": "ACC-789012"
})

Response: {
    "account_id": "ACC-789012",
    "customer_name": "김철수",
    "phone": "010-1234-5678",
    "email": "kim@example.com",
    "account_type": "SAVINGS",
    "account_status": "ACTIVE",
    "creation_date": "2018-03-15",
    "average_monthly_balance": 3000000,
    "credit_score": 720,
    "kyc_status": "VERIFIED",
    "risk_rating": "LOW"
}

→ transaction-db-mcp.get_transaction_history({
    "account_id": "ACC-789012",
    "days": 30
})

Response: [
    {
        "date": "2025-01-17",
        "transaction_count": 3,
        "total_amount": 450000,
        "avg_amount": 150000,
        "max_amount": 200000,
        "unique_countries": 1
    },
    {
        "date": "2025-01-16",
        "transaction_count": 2,
        "total_amount": 320000,
        ...
    }
]

Agent 생각: "이 계좌는 평소 15만원 정도 거래하는데, 갑자기 4500만원을 중국으로 송금했다! 평균의 300배, 매우 의심스럽다."


STEP 4: CRITICAL 거래 상세 지표 분석
→ fraud-detection-mcp.analyze_transaction({
    "transaction": {거래 TXN-2025-001234 상세 정보},
    "account_history": {30일 거래 이력}
})

Response: {
    "transaction_id": "TXN-2025-001234",
    "fraud_probability": 0.94,
    "risk_level": "CRITICAL",
    "recommended_action": "BLOCK_IMMEDIATELY",
    "indicators": [
        {
            "type": "UNUSUAL_AMOUNT",
            "severity": "HIGH",
            "description": "평균 거래액의 300배 초과 (45,000,000원)"
        },
        {
            "type": "HIGH_RISK_COUNTRY",
            "severity": "HIGH",
            "description": "고위험 국가로의 송금: CN"
        },
        {
            "type": "NEW_DEVICE",
            "severity": "MEDIUM",
            "description": "처음 보는 디바이스에서 거래"
        },
        {
            "type": "NO_2FA",
            "severity": "HIGH",
            "description": "2단계 인증 없이 고액 거래"
        }
    ],
    "confidence": 0.94
}

Agent 생각: "확실한 사기 거래다. 즉시 차단 조치!"


STEP 5: CRITICAL 거래 즉시 차단
→ transaction-control-mcp.block_transaction({
    "transaction_id": "TXN-2025-001234",
    "reason": "사기 확률 94% - 비정상적인 고액 해외 송금",
    "block_duration_hours": 72
})

Response: "거래 TXN-2025-001234 차단 완료"

→ transaction-control-mcp.freeze_account({
    "account_id": "ACC-789012",
    "reason": "의심스러운 거래 패턴 - 조사 필요",
    "duration_hours": 24
})

Response: "계좌 ACC-789012 동결 완료 (24시간)"


STEP 6: 고객에게 알림
→ banking-alert-mcp.notify_customer({
    "account_id": "ACC-789012",
    "message": "귀하의 계좌에서 비정상적인 거래가 감지되어 안전을 위해 일시적으로 차단되었습니다.
                본인 거래가 맞으시면 고객센터(1588-XXXX)로 연락 주시기 바랍니다.",
    "channels": ["SMS", "EMAIL", "APP_PUSH"]
})

Response: "고객 알림 발송 완료 (3개 채널)"


STEP 7: 나머지 2건 CRITICAL 거래도 동일하게 처리
→ [Steps 3-6을 TXN-2025-001255, TXN-2025-001289에 반복]

총 CRITICAL 3건 모두 차단 완료


STEP 8: HIGH 위험 거래 8건 검토 대기열 추가
→ transaction-control-mcp.flag_for_review({
    "transaction_id": "TXN-2025-001240",
    "priority": "HIGH",
    "notes": "사기 확률 78% - 수동 검토 필요. 해외 ATM 출금 (금액: 800만원)"
})

→ [나머지 HIGH 7건도 동일하게 처리]

Response: "8건 검토 대기열 추가 완료"


STEP 9: 사기 리포트 생성
→ banking-alert-mcp.generate_fraud_report({
    "time_range_hours": 1,
    "flagged_transactions": [
        {CRITICAL 3건 + HIGH 8건 + MEDIUM 17건 상세 정보}
    ],
    "include_charts": true
})

Response: {
    "report_path": "/tmp/fraud_report_20250118_143000.pdf",
    "summary": {
        "total_flagged": 28,
        "critical": 3,
        "high": 8,
        "medium": 17,
        "blocked_amount": 89000000,  # 8900만원
        "prevented_loss_estimate": 89000000
    }
}


STEP 10: 담당자에게 긴급 알림 발송
→ banking-alert-mcp.send_fraud_alert({
    "recipients": [
        "fraud-team@bank.com",
        "security-manager@bank.com",
        "duty-officer@bank.com"
    ],
    "severity": "CRITICAL",
    "transaction_details": {
        "total_flagged": 28,
        "critical_blocked": 3,
        "high_review": 8,
        "total_amount_blocked": 89000000
    },
    "analysis_result": {
        "time_range": "14:00-15:00",
        "detection_rate": "11.3%",  # 248건 중 28건
        "estimated_loss_prevented": 89000000
    }
})

Response: "알림 발송 완료: 3명에게 전송됨 (Slack, Email)"


STEP 11: 최종 응답
"✅ 실시간 사기 탐지 완료

📊 분석 결과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 분석 기간: 최근 1시간 (14:00-15:00)
• 총 거래 건수: 248건
• 의심 거래 탐지: 28건 (11.3%)

🚨 위험도별 분류:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• CRITICAL (즉시 차단): 3건
  - TXN-2025-001234: 4,500만원 → 중국 송금 ❌ 차단
  - TXN-2025-001255: 2,800만원 → 싱가포르 송금 ❌ 차단
  - TXN-2025-001289: 1,600만원 → 미확인 계좌 ❌ 차단

• HIGH (수동 검토 중): 8건
  - 총 금액: 5,200만원
  - 검토 대기열에 추가됨

• MEDIUM (모니터링 중): 17건

💰 차단된 총 금액: 8,900만원

🛡️ 취한 조치:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 3건 거래 즉시 차단
✅ 3개 계좌 24시간 동결
✅ 고객에게 SMS/이메일/앱푸시 알림 발송
✅ 담당자에게 긴급 알림 (Slack, Email)
✅ 상세 분석 리포트 생성
   → /tmp/fraud_report_20250118_143000.pdf

⏱️ 처리 시간: 2.3초
🎯 예상 손실 방지: 8,900만원

Slack 및 이메일로 상세 리포트가 발송되었습니다!"
"""

"""
ROI 분석:

Before (수동 모니터링):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 사기 탐지팀 3명이 24시간 교대 근무
• 수동 거래 검토: 1건당 평균 3분
• 248건 검토 시간: 12시간 24분
• 휴먼 에러: 의심 거래 20% 놓침
• 대응 시간: 평균 45분 (발견 후 차단까지)
• 인건비: 3명 × 500만원/월 = 1,500만원/월

실제 사기 손실 (월평균):
• 탐지 실패로 인한 손실: 월 2억원
• 늦은 대응으로 인한 추가 손실: 월 5천만원
• 총 손실: 월 2.5억원

After (AI Agent 자동화):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 실시간 자동 분석: 2.3초
• 탐지율: 99.2% (ML 모델 + 규칙 기반)
• 대응 시간: 평균 2초 (즉시 자동 차단)
• False Positive: 5% (수동 검토 필요)
• 인건비: 검토 담당 1명 = 500만원/월

실제 사기 손실 (월평균):
• AI 탐지 후 손실: 월 500만원 (98% 감소)
• 시스템 비용: 월 200만원 (클라우드 + ML)
• 총 비용: 월 700만원

💰 재무적 효과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 사기 손실 감소: 2.5억원 → 500만원/월
• 인건비 절감: 1,500만원 → 500만원/월
• 시스템 비용: 200만원/월

순 절감 효과: 월 2.63억원 (약 300억원/년)

⏱️ 시간적 효과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 거래 검토: 12시간 24분 → 2.3초 (99.99% 단축)
• 사기 대응: 45분 → 2초 (99.93% 단축)
• 담당자 업무량: 95% 감소

📈 추가 효과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 고객 만족도: +40% (빠른 보호)
• 은행 신뢰도: +35%
• 규제 준수: 완벽한 감사 추적
• 직원 만족도: +30% (단순 반복 작업 제거)
"""
```

---

## 6. 회계/세무 (Accounting & Tax): AI 장부 자동화 + OCR 영수증 처리

### 시나리오
**요청**: "이번 달 받은 모든 영수증 사진을 스캔해서 장부에 자동으로 등록하고, 세금 신고 자료 생성해줘"

### 필요한 MCP 서버들

```python
# 1. OCR & Document Analysis MCP
class OCRMCP(Server):
    """OCR + Multimodal LLM을 활용한 문서 분석"""

    def __init__(self):
        super().__init__("ocr-mcp")
        # GPT-4V, Claude 3.5 Sonnet (Vision) 등 사용
        self.llm_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="extract_receipt_info",
                description="영수증/청구서 이미지에서 정보 추출 (OCR + LLM)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "image_url": {"type": "string"},
                        "image_base64": {"type": "string"},
                        "extract_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "추출할 필드 (merchant_name, amount, date, tax, items, etc.)"
                        }
                    }
                }
            ),
            Tool(
                name="extract_invoice_data",
                description="세금계산서/송장 OCR 처리",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "document_path": {"type": "string"},
                        "document_type": {
                            "type": "string",
                            "enum": ["TAX_INVOICE", "INVOICE", "BILL", "RECEIPT"]
                        }
                    },
                    "required": ["document_path"]
                }
            ),
            Tool(
                name="classify_expense",
                description="비용 분류 (접대비, 사무비, 교통비 등)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "merchant_name": {"type": "string"},
                        "items": {"type": "array", "items": {"type": "string"}},
                        "amount": {"type": "number"}
                    },
                    "required": ["merchant_name"]
                }
            ),
            Tool(
                name="verify_document_authenticity",
                description="문서 위조 여부 확인 (Vision AI)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "extract_receipt_info":
            # 이미지 로드
            if 'image_path' in arguments:
                with open(arguments['image_path'], 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
            elif 'image_base64' in arguments:
                image_data = arguments['image_base64']
            else:
                # URL에서 다운로드
                async with httpx.AsyncClient() as client:
                    response = await client.get(arguments['image_url'])
                    image_data = base64.b64encode(response.content).decode('utf-8')

            # Multimodal LLM으로 영수증 분석
            message = self.llm_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": """이 영수증에서 다음 정보를 JSON 형식으로 추출해주세요:

                                {
                                  "merchant_name": "가맹점명",
                                  "merchant_address": "주소",
                                  "business_number": "사업자등록번호",
                                  "date": "거래일시 (YYYY-MM-DD HH:MM:SS)",
                                  "total_amount": 총액 (숫자만),
                                  "tax_amount": 부가세 (숫자만),
                                  "payment_method": "결제수단",
                                  "items": [
                                    {"name": "상품명", "quantity": 수량, "price": 가격}
                                  ],
                                  "category_hint": "비용 분류 힌트 (식사, 교통, 사무용품 등)"
                                }

                                만약 정보가 불명확하면 null로 표시하세요."""
                            }
                        ],
                    }
                ],
            )

            # LLM 응답에서 JSON 추출
            response_text = message.content[0].text

            # JSON 파싱 (LLM이 마크다운 코드 블록으로 감쌀 수 있음)
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                receipt_data = json.loads(json_match.group(1))
            else:
                # 직접 JSON인 경우
                receipt_data = json.loads(response_text)

            return [TextContent(
                type="text",
                text=json.dumps(receipt_data, ensure_ascii=False)
            )]

        elif name == "classify_expense":
            merchant_name = arguments['merchant_name']
            items = arguments.get('items', [])
            amount = arguments.get('amount', 0)

            # LLM으로 비용 분류
            prompt = f"""
            다음 거래를 한국 세법 기준으로 비용 분류해주세요:

            가맹점: {merchant_name}
            구매 항목: {', '.join(items) if items else '정보 없음'}
            금액: {amount:,}원

            분류 옵션:
            - 접대비 (Entertainment)
            - 복리후생비 (Welfare)
            - 여비교통비 (Travel)
            - 차량유지비 (Vehicle)
            - 사무용품비 (Office Supplies)
            - 통신비 (Communication)
            - 수도광열비 (Utilities)
            - 교육훈련비 (Education)
            - 소모품비 (Consumables)
            - 기타 (Other)

            JSON 형식으로 답변:
            {{
              "category": "분류명",
              "confidence": 0.0-1.0,
              "reason": "분류 이유",
              "tax_deductible": true/false,
              "requires_approval": true/false
            }}
            """

            message = self.llm_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            classification = json.loads(response_text)

            return [TextContent(
                type="text",
                text=json.dumps(classification, ensure_ascii=False)
            )]

        elif name == "verify_document_authenticity":
            # Vision AI로 문서 위조 탐지
            image_path = arguments['image_path']

            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            message = self.llm_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": """이 영수증/청구서의 진위 여부를 분석해주세요:

                                확인 항목:
                                1. 이미지 품질 (너무 선명하거나 흐릿한가?)
                                2. 폰트 일관성 (폰트가 일관적인가?)
                                3. 정렬 상태 (텍스트/숫자 정렬이 자연스러운가?)
                                4. 인쇄 흔적 (실제 프린터 출력물인가 vs 포토샵?)
                                5. 그림자/조명 (3D 물체처럼 보이는가?)
                                6. 의심스러운 부분 (숫자 수정, 붙여넣기 흔적 등)

                                JSON 응답:
                                {{
                                  "is_authentic": true/false,
                                  "confidence": 0.0-1.0,
                                  "suspicious_areas": ["의심 영역 설명"],
                                  "recommendation": "APPROVE / MANUAL_REVIEW / REJECT"
                                }}
                                """
                            }
                        ],
                    }
                ]
            )

            verification = json.loads(message.content[0].text)

            return [TextContent(
                type="text",
                text=json.dumps(verification, ensure_ascii=False)
            )]


# 2. Accounting ERP MCP
class AccountingERPMCP(Server):
    """회계 시스템 연동 (ERP)"""

    def __init__(self):
        super().__init__("accounting-erp-mcp")
        self.db_pool = asyncpg.create_pool(
            "postgresql://accounting_user:pass@db:5432/accounting"
        )

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="create_journal_entry",
                description="회계 분개 생성",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "description": {"type": "string"},
                        "entries": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "account_code": {"type": "string"},
                                    "account_name": {"type": "string"},
                                    "debit": {"type": "number"},
                                    "credit": {"type": "number"}
                                }
                            }
                        },
                        "attachments": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["date", "description", "entries"]
                }
            ),
            Tool(
                name="check_duplicate_expense",
                description="중복 비용 처리 확인",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "merchant_name": {"type": "string"},
                        "amount": {"type": "number"},
                        "date": {"type": "string"}
                    },
                    "required": ["amount", "date"]
                }
            ),
            Tool(
                name="generate_tax_report",
                description="세무 신고 자료 생성 (부가세, 소득세)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "report_type": {
                            "type": "string",
                            "enum": ["VAT", "INCOME_TAX", "WITHHOLDING_TAX"]
                        },
                        "year": {"type": "number"},
                        "quarter": {"type": "number"}
                    },
                    "required": ["report_type", "year"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "create_journal_entry":
            date = arguments['date']
            description = arguments['description']
            entries = arguments['entries']
            attachments = arguments.get('attachments', [])

            # 분개 유효성 검사 (차변 = 대변)
            total_debit = sum(e['debit'] for e in entries)
            total_credit = sum(e['credit'] for e in entries)

            if abs(total_debit - total_credit) > 0.01:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"차변({total_debit:,})과 대변({total_credit:,})이 일치하지 않습니다."
                    }, ensure_ascii=False)
                )]

            # DB에 저장
            async with self.db_pool.acquire() as conn:
                # 분개 헤더
                journal_id = await conn.fetchval(
                    """
                    INSERT INTO journal_headers (date, description, created_at)
                    VALUES ($1, $2, NOW())
                    RETURNING journal_id
                    """,
                    date, description
                )

                # 분개 상세
                for entry in entries:
                    await conn.execute(
                        """
                        INSERT INTO journal_entries
                        (journal_id, account_code, account_name, debit, credit)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        journal_id,
                        entry['account_code'],
                        entry['account_name'],
                        entry['debit'],
                        entry['credit']
                    )

                # 첨부파일
                for attachment in attachments:
                    await conn.execute(
                        """
                        INSERT INTO journal_attachments (journal_id, file_path)
                        VALUES ($1, $2)
                        """,
                        journal_id, attachment
                    )

            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "journal_id": journal_id,
                    "total_debit": total_debit,
                    "total_credit": total_credit
                }, ensure_ascii=False)
            )]

        elif name == "check_duplicate_expense":
            amount = arguments['amount']
            date = arguments['date']
            merchant = arguments.get('merchant_name', '')

            async with self.db_pool.acquire() as conn:
                duplicates = await conn.fetch(
                    """
                    SELECT journal_id, description, created_at
                    FROM journal_headers
                    WHERE date = $1
                      AND EXISTS (
                        SELECT 1 FROM journal_entries
                        WHERE journal_entries.journal_id = journal_headers.journal_id
                          AND (debit = $2 OR credit = $2)
                      )
                      AND description ILIKE $3
                    """,
                    date, amount, f"%{merchant}%"
                )

            if duplicates:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "is_duplicate": True,
                        "matches": [dict(d) for d in duplicates]
                    }, default=str, ensure_ascii=False)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({"is_duplicate": False}, ensure_ascii=False)
                )]

        elif name == "generate_tax_report":
            report_type = arguments['report_type']
            year = arguments['year']
            quarter = arguments.get('quarter')

            if report_type == "VAT":
                # 부가가치세 신고서 생성
                if quarter:
                    start_date = f"{year}-{quarter*3-2:02d}-01"
                    end_date = f"{year}-{quarter*3:02d}-31"
                else:
                    start_date = f"{year}-01-01"
                    end_date = f"{year}-12-31"

                async with self.db_pool.acquire() as conn:
                    # 매출세액
                    sales_vat = await conn.fetchval(
                        """
                        SELECT COALESCE(SUM(credit), 0)
                        FROM journal_entries
                        WHERE account_code = '253'  -- 부가세예수금
                          AND date BETWEEN $1 AND $2
                        """,
                        start_date, end_date
                    )

                    # 매입세액
                    purchase_vat = await conn.fetchval(
                        """
                        SELECT COALESCE(SUM(debit), 0)
                        FROM journal_entries
                        WHERE account_code = '136'  -- 부가세대급금
                          AND date BETWEEN $1 AND $2
                        """,
                        start_date, end_date
                    )

                payable_vat = sales_vat - purchase_vat

                report = {
                    "report_type": "부가가치세 신고서",
                    "period": f"{year}년 {quarter}분기" if quarter else f"{year}년",
                    "sales_vat": float(sales_vat),
                    "purchase_vat": float(purchase_vat),
                    "payable_vat": float(payable_vat),
                    "due_date": f"{year}-{quarter*3+1:02d}-25" if quarter else f"{year+1}-01-25"
                }

                return [TextContent(
                    type="text",
                    text=json.dumps(report, ensure_ascii=False)
                )]


# 3. File Storage MCP
class FileStorageMCP(Server):
    """파일 저장 및 관리"""

    def __init__(self):
        super().__init__("file-storage-mcp")
        self.storage_path = "/storage/receipts"

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="list_unprocessed_receipts",
                description="미처리 영수증 목록 조회",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "folder": {"type": "string"}
                    }
                }
            ),
            Tool(
                name="mark_processed",
                description="영수증 처리 완료 표시",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"}
                    },
                    "required": ["file_path"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "list_unprocessed_receipts":
            folder = arguments.get('folder', self.storage_path)

            unprocessed = []
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
                    filepath = os.path.join(folder, filename)
                    # processed 플래그 파일이 없으면 미처리
                    if not os.path.exists(f"{filepath}.processed"):
                        unprocessed.append({
                            "filename": filename,
                            "filepath": filepath,
                            "size": os.path.getsize(filepath),
                            "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                        })

            return [TextContent(
                type="text",
                text=json.dumps(unprocessed, ensure_ascii=False)
            )]

        elif name == "mark_processed":
            file_path = arguments['file_path']
            # .processed 플래그 파일 생성
            with open(f"{file_path}.processed", 'w') as f:
                f.write(datetime.now().isoformat())

            return [TextContent(
                type="text",
                text=json.dumps({"success": True}, ensure_ascii=False)
            )]
```

### Agent 실행 시퀀스

```
사용자: "이번 달 받은 모든 영수증 사진을 스캔해서 장부에 자동으로 등록하고, 세금 신고 자료 생성해줘"

Agent 생각: "회계 자동화 작업이네. 다음 단계로 진행하자:
1. 미처리 영수증 파일 찾기
2. 각 영수증 OCR 처리
3. 비용 분류 및 진위 확인
4. 중복 확인
5. 회계 분개 생성
6. 세금 신고 자료 생성"


STEP 1: 미처리 영수증 찾기
→ file-storage-mcp.list_unprocessed_receipts({
    "folder": "/storage/receipts/2025-01"
})

Response: [
    {
        "filename": "receipt_20250103_starbucks.jpg",
        "filepath": "/storage/receipts/2025-01/receipt_20250103_starbucks.jpg",
        "size": 245678,
        "modified": "2025-01-03T14:30:00"
    },
    {
        "filename": "receipt_20250107_taxi.jpg",
        "filepath": "/storage/receipts/2025-01/receipt_20250107_taxi.jpg",
        "size": 189234,
        "modified": "2025-01-07T09:15:00"
    },
    {
        "filename": "receipt_20250115_office_depot.jpg",
        "filepath": "/storage/receipts/2025-01/receipt_20250115_office_depot.jpg",
        "size": 312456,
        "modified": "2025-01-15T16:45:00"
    },
    ... (총 47건의 미처리 영수증)
]

Agent 생각: "47건의 영수증을 처리해야 하네. 하나씩 OCR 처리하자."


STEP 2: 첫 번째 영수증 OCR 처리
→ ocr-mcp.extract_receipt_info({
    "image_path": "/storage/receipts/2025-01/receipt_20250103_starbucks.jpg"
})

Response: {
    "merchant_name": "스타벅스 강남점",
    "merchant_address": "서울시 강남구 테헤란로 123",
    "business_number": "123-45-67890",
    "date": "2025-01-03 14:23:15",
    "total_amount": 18500,
    "tax_amount": 1682,
    "payment_method": "신용카드",
    "items": [
        {"name": "아메리카노 Tall", "quantity": 2, "price": 8000},
        {"name": "카페라떼 Grande", "quantity": 1, "price": 10500}
    ],
    "category_hint": "식사/음료"
}

Agent 생각: "OCR 성공! 이제 이 비용을 분류하고 진위를 확인해야겠다."


STEP 3: 비용 분류
→ ocr-mcp.classify_expense({
    "merchant_name": "스타벅스 강남점",
    "items": ["아메리카노 Tall", "카페라떼 Grande"],
    "amount": 18500
})

Response: {
    "category": "복리후생비",
    "confidence": 0.85,
    "reason": "사내 직원 음료 구매로 추정. 접대비가 아닌 복리후생비로 분류하는 것이 적절함.",
    "tax_deductible": true,
    "requires_approval": false
}


STEP 4: 문서 진위 확인
→ ocr-mcp.verify_document_authenticity({
    "image_path": "/storage/receipts/2025-01/receipt_20250103_starbucks.jpg"
})

Response: {
    "is_authentic": true,
    "confidence": 0.92,
    "suspicious_areas": [],
    "recommendation": "APPROVE"
}


STEP 5: 중복 확인
→ accounting-erp-mcp.check_duplicate_expense({
    "merchant_name": "스타벅스",
    "amount": 18500,
    "date": "2025-01-03"
})

Response: {
    "is_duplicate": false
}

Agent 생각: "중복 아니고, 진위도 확인됐어. 분개를 생성하자."


STEP 6: 회계 분개 생성
→ accounting-erp-mcp.create_journal_entry({
    "date": "2025-01-03",
    "description": "스타벅스 강남점 - 직원 음료 구매",
    "entries": [
        {
            "account_code": "820",
            "account_name": "복리후생비",
            "debit": 18500,
            "credit": 0
        },
        {
            "account_code": "252",
            "account_name": "미지급금",
            "debit": 0,
            "credit": 18500
        }
    ],
    "attachments": ["/storage/receipts/2025-01/receipt_20250103_starbucks.jpg"]
})

Response: {
    "success": true,
    "journal_id": 2025001234,
    "total_debit": 18500,
    "total_credit": 18500
}


STEP 7: 처리 완료 표시
→ file-storage-mcp.mark_processed({
    "file_path": "/storage/receipts/2025-01/receipt_20250103_starbucks.jpg"
})

Response: {"success": true}

Agent 생각: "첫 번째 영수증 처리 완료! 나머지 46건도 동일하게 처리하자."


STEP 8~53: 나머지 46건 영수증 처리 (병렬 처리)

[각 영수증마다 Steps 2-7 반복]

처리 결과 요약:
- 성공: 44건
- 중복: 2건 (스킵)
  * receipt_20250112_starbucks.jpg → 이미 처리됨
  * receipt_20250118_gs25.jpg → 중복 발견
- 위조 의심: 1건 (수동 검토 필요)
  * receipt_20250120_restaurant.jpg → 폰트 불일치 (Confidence: 0.45)

총 처리 금액: 3,245,800원


STEP 54: 1월 부가가치세 집계
→ accounting-erp-mcp.generate_tax_report({
    "report_type": "VAT",
    "year": 2025,
    "quarter": 1  # 1분기 (1-3월, 1월 데이터만 있음)
})

Response: {
    "report_type": "부가가치세 신고서",
    "period": "2025년 1분기",
    "sales_vat": 8520000,    # 매출세액
    "purchase_vat": 2840000, # 매입세액 (1월 누적)
    "payable_vat": 5680000,  # 납부세액
    "due_date": "2025-04-25"
}


STEP 55: 계정과목별 집계
→ [각 계정과목별로 조회하여 집계]

결과:
- 복리후생비 (820): 450,000원
- 여비교통비 (810): 890,000원
- 접대비 (800): 1,200,000원
- 사무용품비 (730): 380,000원
- 통신비 (740): 325,800원
총 44건, 3,245,800원


STEP 56: 최종 응답
"✅ 1월 영수증 자동 처리 완료!

📊 처리 결과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 총 영수증 수: 47건
• ✅ 성공 처리: 44건
• ⚠️  중복 발견: 2건 (스킵)
  - receipt_20250112_starbucks.jpg
  - receipt_20250118_gs25.jpg
• 🚨 수동 검토 필요: 1건 (위조 의심)
  - receipt_20250120_restaurant.jpg
    → 폰트 불일치, Confidence: 0.45

💰 금액 집계:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 총 비용: 3,245,800원

계정과목별 분류:
  - 복리후생비: 450,000원 (14%)
  - 여비교통비: 890,000원 (27%)
  - 접대비: 1,200,000원 (37%)
  - 사무용품비: 380,000원 (12%)
  - 통신비: 325,800원 (10%)

📝 생성된 분개: 44건
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
모든 영수증이 회계 시스템에 자동 등록되었습니다.
첨부파일과 함께 저장되어 감사 추적이 가능합니다.

🧾 세금 신고 자료:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[2025년 1분기 부가가치세]
• 매출세액: 8,520,000원
• 매입세액: 2,840,000원 (1월 누적)
• 납부세액: 5,680,000원
• 신고 기한: 2025-04-25

⚠️  수동 검토 필요 항목:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. receipt_20250120_restaurant.jpg
   → 위조 의심 (폰트 불일치, Confidence: 0.45)
   → 파일 경로: /storage/receipts/2025-01/receipt_20250120_restaurant.jpg
   → 담당자 확인 필요

📎 첨부 파일:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 전체 분개 장부: /reports/journal_entries_202501.xlsx
• 세무 신고 자료: /reports/vat_report_2025_Q1.pdf
• 비용 분석 차트: /reports/expense_analysis_202501.png

⏱️ 처리 시간: 23초 (영수증 47건)
💾 모든 데이터는 회계 시스템에 저장되었습니다."
"""

"""
ROI 분석:

Before (수동 처리):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 회계 담당자가 영수증 하나하나 수기 입력
• 1건당 평균 시간:
  - 영수증 확인: 30초
  - 금액/날짜 입력: 1분
  - 계정과목 분류: 30초 (판단 필요)
  - 분개 작성: 1분
  - ERP 입력: 30초
  → 총 3분 30초/건

• 47건 처리: 164.5분 (2시간 44분)
• 입력 오류율: 5% (금액, 날짜 오타, 분류 실수)
• 중복 처리: 월 2-3건 발생 (수동으로 찾기 어려움)
• 위조 영수증 탐지: 거의 불가능 (육안으로 판단)

• 월말 세무 자료 준비:
  - 계정과목별 집계: 30분
  - 부가세 계산: 20분
  - 리포트 작성: 40분
  - 검증 및 수정: 20분
  → 총 110분 (1시간 50분)

월간 총 시간 (영수증 200건 기준):
• 입력: 200건 × 3.5분 = 700분 (11시간 40분)
• 세무 자료: 110분 (1시간 50분)
• 총: 810분 (13시간 30분)

인건비: 13.5시간 × 50,000원/시간 = 675,000원/월

After (AI Agent + OCR):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• OCR + Multimodal LLM 자동 처리:
  - 영수증 읽기: 2초/건 (Vision AI)
  - 자동 분류: 0.5초/건 (LLM 분석)
  - 위조 탐지: 1초/건 (Vision AI)
  - 분개 생성: 0.5초/건
  - DB 저장: 0.3초/건
  → 총 4.3초/건

• 47건 처리: 23초 (병렬 처리 가속)
• 200건 처리: 약 90초 (1.5분)

• 입력 오류율: 0.1% (OCR 인식 오류만, 자동 검증으로 최소화)
• 중복 자동 탐지: 100% (DB 조회로 즉시 발견)
• 위조 영수증 탐지: 92% 정확도 (Vision AI 분석)

• 월말 세무 자료:
  - 자동 집계: 3초
  - 자동 계산: 1초
  - 자동 리포트: 5초
  → 총 9초

월간 총 시간 (영수증 200건 기준):
• 입력: 1.5분
• 세무 자료: 9초
• 수동 검토: 10분 (위조 의심 건만)
• 총: 약 12분

인건비: 0.2시간 × 50,000원 = 10,000원/월

💰 재무적 효과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 인건비 절감:
  - Before: 675,000원/월
  - After: 10,000원/월 (수동 검토만)
  - 절감: 665,000원/월 (약 798만원/년)

• OCR API 비용:
  - Claude Vision API: 영수증당 약 100원
  - 200건 × 100원 = 20,000원/월
  - 연간: 240,000원

• 시스템 운영 비용:
  - 서버 비용: 월 30,000원 (클라우드 GPU)
  - 저장소: 월 10,000원
  - 총: 40,000원/월 (480,000원/년)

• 순 절감:
  - 월: 665,000 - 20,000 - 40,000 = 605,000원
  - 연: 7,260,000원

⏱️ 시간적 효과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 월간 처리 시간: 13.5시간 → 12분 (99.1% 단축)
• 회계 담당자는 전략적 업무에 집중
  - 재무 분석
  - 예산 관리
  - 세무 전략 수립
• 월말 마감: 반나절 → 10분

🎯 품질 효과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 입력 오류: 5% → 0.1% (50배 감소)
• 중복 처리 방지: 100% (자동 감지)
• 위조 영수증 탐지: 0% → 92% (새로운 기능)
• 세무 조사 대응: 완벽한 감사 추적
  - 모든 영수증 이미지 첨부
  - 처리 이력 자동 기록
  - 분개 자동 검증

📈 추가 효과:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 실시간 비용 모니터링 가능
  - 일별/주별/월별 지출 트렌드
  - 계정과목별 실시간 집계
  - 예산 대비 실적 자동 분석
• 이상 지출 즉시 감지
  - 평균의 300% 초과 시 알림
  - 특정 카테고리 급증 감지
• CFO에게 실시간 대시보드 제공
• 연말 세무 조사 대비 완벽
  - 모든 증빙 디지털화
  - 자동 감사 추적
"""
```

---

## 7. 법률사무소 (Law Firm): AI 계약서 분석 + 판례 검색

### 시나리오
**요청**: "이 공급계약서 PDF를 분석해서 리스크 있는 조항 찾고, 관련 판례 검색해서 클라이언트에게 리포트 보내줘"

### 필요한 MCP 서버들

```python
# 1. Legal Document OCR MCP
class LegalDocumentOCRMCP(Server):
    """법률 문서 OCR + 분석"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="extract_contract_text",
                description="계약서 PDF에서 텍스트 추출 (OCR + 구조화)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pdf_path": {"type": "string"},
                        "extract_clauses": {"type": "boolean"}
                    }
                }
            ),
            Tool(
                name="analyze_contract_risks",
                description="계약서 리스크 분석 (LLM)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "contract_text": {"type": "string"},
                        "contract_type": {"type": "string"}
                    }
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "analyze_contract_risks":
            contract_text = arguments['contract_text']

            prompt = f"""
            다음 계약서를 분석하여 법적 리스크를 찾아주세요:

            {contract_text}

            분석 항목:
            1. 불공정 조항 (일방적 해지권, 과도한 손해배상 등)
            2. 불명확한 조항 (해석 여지가 많은 문구)
            3. 법률 위반 가능성
            4. 누락된 필수 조항
            5. 상대방에게 유리한 조항

            JSON 응답:
            {{
              "risk_clauses": [
                {{
                  "clause_number": "제X조",
                  "clause_text": "조항 내용",
                  "risk_level": "HIGH/MEDIUM/LOW",
                  "risk_type": "리스크 유형",
                  "description": "상세 설명",
                  "recommendation": "수정 권고사항"
                }}
              ],
              "overall_risk_score": 0-100,
              "missing_clauses": ["누락된 조항 목록"]
            }}
            """

            # LLM 호출 (생략)
            return [TextContent(type="text", text="...")]


# 2. Legal Database MCP
class LegalDatabaseMCP(Server):
    """판례 검색 + 법률 DB"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="search_case_law",
                description="판례 검색",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "court": {"type": "string"},
                        "year_range": {"type": "object"}
                    }
                }
            ),
            Tool(
                name="search_statutes",
                description="법률/조문 검색",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "statute_name": {"type": "string"},
                        "article_number": {"type": "string"}
                    }
                }
            )
        ]


# 3. Document Generation MCP
class LegalDocumentGenMCP(Server):
    """법률 문서 생성"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="generate_legal_opinion",
                description="법률 의견서 생성",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "case_summary": {"type": "string"},
                        "analysis": {"type": "object"},
                        "precedents": {"type": "array"}
                    }
                }
            )
        ]
```

### Agent 실행 시퀀스 (요약)

```
사용자: "이 공급계약서 PDF를 분석해서 리스크 있는 조항 찾고, 관련 판례 검색해서 리포트 보내줘"

STEP 1: PDF OCR 처리 → 계약서 텍스트 추출 (15개 조항)
STEP 2: LLM 리스크 분석 → 7개 리스크 조항 발견
  - HIGH: 3개 (일방적 해지권, 과도한 손해배상, 경쟁 금지)
  - MEDIUM: 4개
STEP 3: 각 리스크별 판례 검색 → 23개 관련 판례 발견
STEP 4: 법률 의견서 자동 생성 (PDF 20페이지)
STEP 5: 클라이언트에게 이메일 발송

ROI:
- 변호사 수동 검토: 4시간 → AI: 3분 (98.8% 단축)
- 판례 검색: 1.5시간 → 15초
- 의견서 작성: 2시간 → 2분
- 청구 가능 시간 증가: 월 80시간 (2000만원 추가 매출)
```

---

## 8. 학교/교육 (Education): AI 자동 채점 + 학습 분석

### 시나리오
**요청**: "중간고사 수학 시험지 150장을 스캔해서 자동 채점하고, 학생별 취약 영역 분석해줘"

### 필요한 MCP 서버들

```python
# 1. Exam Grading OCR MCP
class ExamGradingOCRMCP(Server):
    """시험지 OCR + 자동 채점"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="scan_exam_paper",
                description="시험지 스캔 (학생 답안 OCR)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "exam_type": {"type": "string", "enum": ["MULTIPLE_CHOICE", "SHORT_ANSWER", "ESSAY"]}
                    }
                }
            ),
            Tool(
                name="grade_math_answer",
                description="수학 답안 채점 (Vision AI + 수식 인식)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "student_answer": {"type": "string"},
                        "correct_answer": {"type": "string"},
                        "question_type": {"type": "string"},
                        "partial_credit": {"type": "boolean"}
                    }
                }
            ),
            Tool(
                name="analyze_handwriting",
                description="필기체 인식 (학생 이름, 답안)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_region": {"type": "object"}
                    }
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "grade_math_answer":
            student_ans = arguments['student_answer']
            correct_ans = arguments['correct_answer']

            # Multimodal LLM으로 수학 답안 평가
            prompt = f"""
            수학 문제 채점:

            정답: {correct_ans}
            학생 답안: {student_ans}

            다음을 평가해주세요:
            1. 답이 정확한가?
            2. 풀이 과정이 올바른가?
            3. 부분 점수 부여 가능한가?

            JSON 응답:
            {{
              "is_correct": true/false,
              "score": 0-10,
              "feedback": "채점 피드백",
              "mistakes": ["실수한 부분"]
            }}
            """

            # LLM 호출
            return [TextContent(type="text", text="...")]


# 2. Learning Analytics MCP
class LearningAnalyticsMCP(Server):
    """학습 분석 + 리포트"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="analyze_weak_areas",
                description="학생별 취약 영역 분석",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "student_id": {"type": "string"},
                        "exam_results": {"type": "array"}
                    }
                }
            ),
            Tool(
                name="generate_study_plan",
                description="맞춤 학습 계획 생성",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "student_id": {"type": "string"},
                        "weak_areas": {"type": "array"}
                    }
                }
            )
        ]
```

### Agent 실행 시퀀스 (요약)

```
사용자: "중간고사 수학 시험지 150장을 스캔해서 자동 채점하고, 학생별 취약 영역 분석해줘"

STEP 1: 시험지 150장 OCR 처리
  → 학생 이름, 학번 인식 (필기체)
  → 각 문제별 답안 추출 (객관식 + 주관식)

STEP 2: 자동 채점 (Vision AI)
  - 객관식 30문제: OMR 자동 인식
  - 서술형 10문제: LLM 기반 채점
    * 수식 인식
    * 풀이 과정 평가
    * 부분 점수 자동 부여

STEP 3: 학생별 성적 집계
  - 평균: 73.5점
  - 최고점: 98점
  - 최저점: 42점

STEP 4: 문항별 정답률 분석
  - 17번 문제: 정답률 23% (가장 어려움)
  - 5번 문제: 정답률 94% (가장 쉬움)

STEP 5: 학생별 취약 영역 분석 (150명)
  예: 김철수 학생
    - 약한 영역: 이차함수 (40%), 확률 (55%)
    - 강한 영역: 방정식 (90%), 도형 (85%)

STEP 6: 맞춤 학습 계획 생성 (150명)
STEP 7: 학부모에게 리포트 자동 발송

처리 시간: 8분 (150명)

ROI:
- 교사 수동 채점: 12시간 → AI: 8분 (99.0% 단축)
- 객관성: 주관 편향 제거, 일관된 기준
- 피드백 속도: 3일 → 즉시
- 교사는 교육 본질에 집중 가능
```

---

## 9. 제조업 (Manufacturing): AI 품질 검사 + Vision AI

### 시나리오
**요청**: "오늘 생산된 전자부품 5000개를 카메라로 촬영해서 불량품 자동 감지하고, 불량 유형별 리포트 생성해줘"

### 필요한 MCP 서버들

```python
# 1. Vision Inspection MCP
class VisionInspectionMCP(Server):
    """Vision AI 품질 검사"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="detect_defects",
                description="제품 결함 탐지 (Vision AI)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "product_type": {"type": "string"},
                        "inspection_mode": {"type": "string", "enum": ["SURFACE", "DIMENSION", "COLOR", "ALL"]}
                    }
                }
            ),
            Tool(
                name="measure_dimensions",
                description="치수 측정 (Computer Vision)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"},
                        "calibration_ratio": {"type": "number"}
                    }
                }
            ),
            Tool(
                name="classify_defect_type",
                description="불량 유형 분류",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "defect_image": {"type": "string"}
                    }
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        if name == "detect_defects":
            image_path = arguments['image_path']

            # Vision AI로 결함 탐지
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            prompt = """
            이 전자부품 이미지를 분석하여 결함을 찾아주세요:

            검사 항목:
            1. 표면 스크래치
            2. 변색
            3. 찌그러짐
            4. 이물질 부착
            5. 치수 이상
            6. 납땜 불량

            JSON 응답:
            {
              "has_defect": true/false,
              "defects": [
                {
                  "type": "결함 유형",
                  "severity": "CRITICAL/MAJOR/MINOR",
                  "location": {"x": 0, "y": 0, "w": 0, "h": 0},
                  "confidence": 0.0-1.0
                }
              ],
              "quality_grade": "A/B/C/D/F"
            }
            """

            # Multimodal LLM 호출
            return [TextContent(type="text", text="...")]


# 2. Manufacturing MES MCP
class ManufacturingMESMCP(Server):
    """제조 실행 시스템 연동"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="log_inspection_result",
                description="검사 결과 MES에 기록",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "inspection_result": {"type": "object"},
                        "action": {"type": "string", "enum": ["PASS", "REWORK", "SCRAP"]}
                    }
                }
            ),
            Tool(
                name="generate_quality_report",
                description="품질 리포트 생성",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "product_line": {"type": "string"}
                    }
                }
            )
        ]
```

### Agent 실행 시퀀스 (요약)

```
사용자: "오늘 생산된 전자부품 5000개를 카메라로 촬영해서 불량품 자동 감지하고, 불량 유형별 리포트 생성해줘"

STEP 1: 5000개 제품 이미지 로드 (생산 라인 카메라)
STEP 2: Vision AI 병렬 검사 (GPU 가속)
  → 처리 속도: 초당 50개
  → 총 처리 시간: 100초 (1분 40초)

STEP 3: 검사 결과 집계
  - 합격 (PASS): 4,732개 (94.6%)
  - 재작업 (REWORK): 189개 (3.8%)
  - 폐기 (SCRAP): 79개 (1.6%)

STEP 4: 불량 유형별 분석
  - 표면 스크래치: 98건 (36.6%)
  - 납땜 불량: 67건 (25.0%)
  - 치수 이상: 48건 (17.9%)
  - 변색: 32건 (11.9%)
  - 기타: 23건 (8.6%)

STEP 5: 불량 원인 분석 (AI)
  - 스크래치 → 컨베이어 벨트 거칠음 (교체 필요)
  - 납땜 불량 → 온도 설정 불안정 (보정 필요)

STEP 6: 품질 리포트 생성 + 대시보드 업데이트
STEP 7: 생산 관리자에게 알림 발송

ROI:
- 육안 검사: 2명 × 4시간 = 8시간 → AI: 1.7분 (99.6% 단축)
- 검사 일관성: 사람 85% → AI 99.2%
- 불량 탐지율: 육안 90% → Vision AI 98.5%
- 연간 불량 손실 감소: 3억원 → 3천만원 (90% 감소)
```

---

## 10. 의료/헬스케어 (Healthcare): AI 의료 기록 분석 + 처방전 OCR

### 시나리오
**요청**: "환자 차트 PDF와 처방전 이미지를 분석해서 약물 상호작용 체크하고, 전자 의무 기록에 자동 입력해줘"

### 필요한 MCP 서버들

```python
# 1. Medical OCR MCP
class MedicalOCRMCP(Server):
    """의료 문서 OCR (HIPAA 준수)"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="extract_prescription",
                description="처방전 OCR (필기체 인식)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    }
                }
            ),
            Tool(
                name="parse_medical_chart",
                description="환자 차트 파싱",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pdf_path": {"type": "string"}
                    }
                }
            )
        ]


# 2. Drug Interaction MCP
class DrugInteractionMCP(Server):
    """약물 상호작용 체크"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="check_drug_interactions",
                description="약물 상호작용 확인",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "medications": {"type": "array", "items": {"type": "string"}},
                        "patient_conditions": {"type": "array"}
                    }
                }
            )
        ]


# 3. EHR Integration MCP
class EHRIntegrationMCP(Server):
    """전자 의무 기록 시스템 연동"""

    async def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="update_patient_record",
                description="환자 기록 업데이트",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string"},
                        "record_data": {"type": "object"}
                    }
                }
            )
        ]
```

### Agent 실행 시퀀스 (요약)

```
사용자: "환자 차트 PDF와 처방전 이미지를 분석해서 약물 상호작용 체크하고, EHR에 자동 입력해줘"

STEP 1: 처방전 OCR (의사 필기체)
  → 약물 5개 인식:
    1. Aspirin 100mg
    2. Lisinopril 10mg
    3. Metformin 500mg
    4. Atorvastatin 20mg
    5. Warfarin 5mg

STEP 2: 환자 차트 파싱
  → 기존 복용약: Warfarin, Insulin
  → 알레르기: Penicillin
  → 진단명: 당뇨, 고혈압, 협심증

STEP 3: 약물 상호작용 체크
  ⚠️ CRITICAL 발견:
    - Aspirin + Warfarin → 출혈 위험 증가 (심각)
    - Lisinopril + Aspirin → 신장 기능 저하 가능

STEP 4: 의사에게 즉시 알림
  "CRITICAL: 약물 상호작용 경고! 처방 재검토 필요"

STEP 5: 대체 약물 제안
  - Aspirin 대신 → Clopidogrel 권장

STEP 6: 수정된 처방으로 EHR 업데이트

ROI:
- 약물 상호작용 체크: 수동 3분 → AI 5초
- 의료 사고 방지: 연간 예상 2건 → 0건
- 약사 업무 효율: +40%
- 환자 안전: 크게 향상
```

---

## 11. 부동산 (Real Estate): AI 매물 분석 + 이미지 인식

### 시나리오
**요청**: "새로 등록된 아파트 매물 사진 50장을 분석해서 방 구조, 컨디션, 예상 가격 산정하고 고객 매칭해줘"

### 핵심 MCP: Property Vision MCP (Vision AI로 부동산 사진 분석), Valuation MCP (가격 산정), Customer Matching MCP

```
Agent 실행:
1. 50개 매물 사진 Vision AI 분석
   - 방 개수, 구조 자동 인식
   - 인테리어 상태 평가 (A~F등급)
   - 채광, 뷰, 마감재 품질 자동 판정
2. 시세 데이터 + AI로 예상 가격 산정
3. 고객 선호도 DB와 매칭 (20명 매칭됨)
4. 개인화된 매물 추천 이메일 자동 발송

ROI:
- 매물 분석: 50개 × 10분 = 8시간 → AI: 3분 (99.4% 단축)
- 가격 산정 정확도: ±15% → ±5%
- 고객 매칭 시간: 2시간 → 30초
- 계약 성사율: +25% 향상
```

---

## 12. 소매/유통 (Retail): AI 재고 관리 + 수요 예측

### 시나리오
**요청**: "전국 50개 매장의 재고 데이터와 판매 이력을 분석해서 다음 주 발주량 최적화하고, 품절 위험 상품 알려줘"

### 핵심 MCP: Inventory MCP, Sales Forecasting MCP (시계열 ML), Supply Chain MCP

```
Agent 실행:
1. 50개 매장 × 5,000개 상품 = 25만개 SKU 분석
2. 시계열 ML로 다음 주 수요 예측
   - 계절성, 트렌드, 프로모션 효과 반영
   - 지역별 특성 고려
3. 재고 최적화 알고리즘 실행
4. 품절 위험 HIGH 상품 123개 발견
5. 자동 발주 제안 생성 (상품별 최적 수량)

ROI:
- 재고 분석: 3명 × 6시간 = 18시간 → AI: 5분
- 재고 비용: 15% 감소 (과잉 재고 방지)
- 품절로 인한 매출 손실: 80% 감소
- 연간 절감: 8억원
```

---

## 13. 물류/배송 (Logistics): AI 배송 경로 최적화 + 실시간 추적

### 시나리오
**요청**: "오늘 배송할 주문 850건을 차량 25대에 최적 배분하고, 실시간 교통 상황 반영해서 경로 업데이트해줘"

### 핵심 MCP: Route Optimization MCP (VRP 알고리즘), Traffic MCP (실시간 교통), Delivery Tracking MCP

```
Agent 실행:
1. 850건 주문을 25대 차량에 최적 배분 (VRP 알고리즘)
   - 차량 용량, 배송 시간대, 우선순위 고려
2. 각 차량별 최적 경로 계산 (Google Maps API)
3. 실시간 교통 상황 모니터링
   - 서울 강남 교통 체증 발생 → 자동 우회 경로 제안
4. 고객에게 실시간 배송 알림 발송
5. 예상 배송 완료: 오후 6시 23분

ROI:
- 경로 계획: 2시간 → 3분 (97.5% 단축)
- 총 이동 거리: 2,340km → 1,890km (19% 감소)
- 연료비 절감: 월 450만원
- 배송 시간 준수율: 82% → 96%
```

---

## 14. HR/채용 (Human Resources): AI 이력서 분석 + 후보자 매칭

### 시나리오
**요청**: "시니어 백엔드 개발자 채용 공고에 들어온 이력서 320장을 OCR로 파싱하고, 요구사항 매칭해서 상위 20명 추천해줘"

### 핵심 MCP: Resume OCR MCP (이력서 파싱), Candidate Matching MCP (LLM 기반 매칭), Interview Scheduling MCP

```
Agent 실행:
1. 이력서 320장 OCR 처리
   - PDF, 워드, 이미지 모두 지원
   - 학력, 경력, 기술 스택 자동 추출
2. LLM으로 JD(Job Description)와 매칭
   필수 요구사항:
   - Python, Django 5년+ ✓
   - AWS 경험 ✓
   - 대규모 트래픽 처리 경험 ✓
3. 320명 중 상위 20명 선발 (매칭 점수 85점 이상)
4. 자동으로 1차 면접 일정 제안 이메일 발송
5. 면접관에게 후보자 요약 리포트 전달

ROI:
- 이력서 검토: 320장 × 5분 = 27시간 → AI: 4분 (99.8% 단축)
- 채용 기간: 45일 → 18일 (60% 단축)
- 채용 담당자는 인터뷰에만 집중
- 채용 품질: +30% 향상 (더 정확한 매칭)
```

---

## 15. 마케팅 (Marketing): AI 광고 성과 분석 + 크리에이티브 추천

### 시나리오
**요청**: "지난 달 실행한 페이스북/구글 광고 캠페인 50개를 분석하고, 성과 좋은 크리에이티브 패턴 찾아서 다음 캠페인 추천해줘"

### 핵심 MCP: Ads Analytics MCP, Creative Analysis MCP (Vision AI로 이미지 분석), Recommendation MCP

```
Agent 실행:
1. 50개 캠페인 데이터 수집 (Facebook Ads API, Google Ads API)
   - 노출, 클릭, 전환, CPA, ROAS
2. Vision AI로 크리에이티브 이미지 분석
   - 색상 팔레트, 레이아웃, 텍스트 위치, 모델 포즈 등
3. 성과와 크리에이티브 요소 상관관계 분석
   발견:
   - 파란색 배경 → CТR +35%
   - 왼쪽 정렬 텍스트 → 전환율 +22%
   - 인물 정면 응시 → 참여도 +40%
4. LLM으로 다음 캠페인 크리에이티브 가이드 생성
5. 예상 성과 시뮬레이션

ROI:
- 분석 시간: 8시간 → 5분 (98.9% 단축)
- 캠페인 ROAS: 2.3 → 4.1 (78% 향상)
- 마케팅 비용 효율: +65%
- 월 광고 예산 2억원 → 4.1억원 매출 (기존 2.3억원)
```

---

## 16. 예술/크리에이티브 (Arts & Creative): AI 아트 큐레이션 + 스타일 분석

### 시나리오
**요청**: "갤러리에 전시할 작품 후보 500점의 이미지를 분석해서 스타일별로 분류하고, 관람객 선호도 예측해서 전시 큐레이션 제안해줘"

### 핵심 MCP: Art Analysis MCP (Vision AI 스타일 인식), Curation MCP, Visitor Preference MCP

```
Agent 실행:
1. 작품 500점 Vision AI 분석
   - 화풍: 인상주의 89점, 추상 142점, 사실주의 73점...
   - 색조: 따뜻한 색 234점, 차가운 색 198점...
   - 주제: 풍경 156점, 인물 89점, 정물 67점...
   - 시대: 19세기 23점, 현대 477점
2. 관람객 과거 선호도 데이터 분석 (30-40대 여성 다수)
3. AI 큐레이션 제안:
   섹션 1: "도시의 색채" - 현대 추상 작품 45점
   섹션 2: "자연의 순간" - 인상주의 풍경 32점
   섹션 3: "인간의 표정" - 현대 인물화 28점
4. 동선 최적화 (관람 시간 45분 예상)
5. 작품 설명 자동 생성 (LLM)

ROI:
- 큐레이션 시간: 큐레이터 2주 → AI: 1시간
- 관람객 만족도: +45% (AI 기반 선호도 반영)
- 전시 회전율: +30% (더 빠른 기획)
- 작품 판매율: +18%
```

---

## 17. 보험 (Insurance): AI 사고 접수 + 자동 심사

### 시나리오
**요청**: "자동차 사고 접수 사진 15장을 분석해서 파손 정도 평가하고, 보험금 자동 산정해줘"

### 핵심 MCP: Damage Assessment MCP (Vision AI), Insurance Claim MCP, Fraud Detection MCP

```
Agent 실행:
1. 사고 차량 사진 15장 Vision AI 분석
   - 파손 부위: 앞범퍼, 헤드라이트, 보닛
   - 손상 정도: 중파 (Medium)
   - 예상 수리 비용: 320만원
2. 사고 경위서 OCR 처리
3. 과거 유사 사례 검색 (23건 발견)
4. Fraud Detection: 이상 징후 없음 (정상 사고)
5. 자동 보험금 산정: 285만원 (자기부담금 35만원 제외)
6. 고객에게 SMS 발송: "심사 완료, 285만원 지급 예정"

ROI:
- 사고 심사: 45분 → 2분 (95.6% 단축)
- 보험 사기 탐지: +40% 향상
- 고객 만족도: +50% (빠른 처리)
- 심사 인력: 30명 → 8명
```

---

## 18. 식음료/레스토랑 (F&B): AI 메뉴 추천 + 재고 최적화

### 시나리오
**요청**: "지난 3개월 주문 데이터를 분석해서 다음 주 식자재 발주 최적화하고, 고객별 메뉴 추천 시스템 만들어줘"

### 핵심 MCP: Menu Analytics MCP, Inventory Forecasting MCP, Personalization MCP

```
Agent 실행:
1. 3개월 주문 데이터 분석 (12,450건)
   - 인기 메뉴 TOP 10
   - 요일별/시간대별 패턴
   - 날씨 상관관계
2. 다음 주 수요 예측
   - 월: 비 예보 → 찌개류 +30%
   - 금: 급여일 → 고가 메뉴 +45%
3. 식자재 최적 발주량 계산
   - 소고기 65kg, 돼지고기 48kg...
   - 폐기 손실 최소화
4. 고객별 메뉴 추천 (앱 푸시)
   예: "김철수님, 오늘은 부대찌개 어때요? (과거 선호 메뉴 기반)"

ROI:
- 식자재 폐기: 월 8% → 2% (75% 감소)
- 재고 비용 절감: 월 380만원
- 주문 단가: +15% (개인화 추천 효과)
- 고객 재방문율: +28%
```

---

## 19. 여행/관광 (Travel & Tourism): AI 여행 플래너 + 실시간 최적화

### 시나리오
**요청**: "유럽 14일 여행 계획을 세워줘. 예산 500만원, 관심사는 역사/미술/맛집이야"

### 핵심 MCP: Travel Planning MCP, Booking MCP (항공/호텔 API), Weather MCP, Review Analysis MCP

```
Agent 실행:
1. 예산 500만원 최적 배분
   - 항공: 120만원
   - 숙박: 180만원 (14박)
   - 식사/관광: 200만원
2. 관심사 기반 도시 선정
   - 파리(4일) → 피렌체(3일) → 로마(4일) → 바르셀로나(3일)
3. 일정별 관광지 추천 (LLM + 리뷰 분석)
   Day 1 파리: 루브르 → 오르세 → 에펠탑
   Day 2 파리: 베르사유 → 생트샤펠 → 몽마르트
4. 맛집 추천 (Google Maps 리뷰 AI 분석)
5. 실시간 항공권/호텔 가격 모니터링
   → 로마 호텔 20% 할인 발견! 자동 예약 제안
6. 날씨 예보 반영 일정 조정

ROI:
- 여행 계획: 수동 8시간 → AI: 3분
- 비용 최적화: 30만원 절감
- 만족도: 맞춤 추천으로 +60% 향상
```

---

## 20. 미디어/방송 (Media & Broadcasting): AI 영상 편집 + 자막 생성

### 시나리오
**요청**: "2시간짜리 인터뷰 영상을 하이라이트 3분으로 자동 편집하고, 한/영 자막 생성해줘"

### 핵심 MCP: Video Analysis MCP (Vision + Audio AI), Subtitle MCP (음성 인식 + 번역), Editing MCP

```
Agent 실행:
1. 2시간 영상 AI 분석
   - 음성 인식 → 텍스트 변환
   - 화제별 세그먼트 분할 (12개 주제)
   - 감정 분석 (흥미도, 중요도 점수화)
2. 하이라이트 자동 추출 (상위 3분)
   - 가장 흥미로운 발언 5개
   - 웃음 소리 많은 구간 3개
   - 핵심 메시지 2개
3. 자동 편집 (컷, 트랜지션, BGM 추가)
4. 한국어 자막 자동 생성 (음성 인식 정확도 98%)
5. 영어 자막 생성 (AI 번역)
6. 썸네일 자동 생성 (가장 인상적인 장면)

ROI:
- 편집 시간: 6시간 → 8분 (98% 단축)
- 자막 작성: 4시간 → 2분
- 편집자는 크리에이티브 작업에 집중
- 영상 제작 처리량: 3배 증가
```

---

## 🎯 산업별 MCP + Agent 적용 요약

```
┌────────────────┬──────────────────────┬──────────────┬─────────────┐
│   산업군       │  핵심 MCP             │  시간 절감   │  주요 효과  │
├────────────────┼──────────────────────┼──────────────┼─────────────┤
│ 금융/은행      │ Fraud Detection      │ 99.99%       │ 손실 -98%   │
│ 회계/세무      │ OCR + ERP            │ 99.7%        │ 오류 -98%   │
│ 법률사무소     │ Legal AI + 판례      │ 98.8%        │ 매출 +40%   │
│ 학교/교육      │ Exam Grading OCR     │ 99.0%        │ 피드백 즉시 │
│ 제조업         │ Vision Inspection    │ 99.6%        │ 불량 -90%   │
│ 의료/병원      │ Medical OCR + Drug   │ 95%          │ 사고 방지   │
│ 부동산         │ Property Vision AI   │ 99.4%        │ 성사율 +25% │
│ 소매/유통      │ Demand Forecasting   │ 97%          │ 절감 8억    │
│ 물류/배송      │ Route Optimization   │ 97.5%        │ 연료 -19%   │
│ HR/채용        │ Resume OCR + Match   │ 99.8%        │ 기간 -60%   │
│ 마케팅         │ Ads Analytics + AI   │ 98.9%        │ ROAS +78%   │
│ 예술/갤러리    │ Art Curation AI      │ 95%          │ 판매 +18%   │
│ 보험           │ Damage Assessment    │ 95.6%        │ 만족 +50%   │
│ 식음료         │ Menu Optimization    │ 90%          │ 폐기 -75%   │
│ 여행/관광      │ Travel Planning      │ 98%          │ 만족 +60%   │
│ 미디어/방송    │ Video AI Editing     │ 98%          │ 생산 3배    │
└────────────────┴──────────────────────┴──────────────┴─────────────┘

**공통 패턴:**
1. OCR/Vision AI → 문서/이미지 자동 처리
2. LLM → 분석, 분류, 생성
3. Specialized API → 도메인 지식
4. Automation → 워크플로우 자동화
5. Real-time → 즉각적인 의사결정

**ROI 평균:**
- 시간 절감: 95-99%
- 비용 절감: 60-90%
- 품질 향상: 30-50%
- 매출 증대: 20-80%

**성공 요인:**
- 여러 MCP 서버의 조합 (마이크로서비스 아키텍처)
- AI Agent의 지능적 오케스트레이션
- 도메인 특화 MCP 개발
- 실시간 데이터 연동
- 인간과 AI의 협업 (Human-in-the-loop)
```

---

## 🚀 MCP + Agent 아키텍처 구현 가이드

### 1. MCP 서버 개발 단계

```python
# Step 1: MCP 서버 기본 구조
from mcp.server import Server
from mcp.types import Tool, TextContent
from typing import List, Dict

class YourCustomMCP(Server):
    def __init__(self):
        super().__init__("your-mcp-name")
        # 초기화 (DB 연결, API 클라이언트 등)

    async def list_tools(self) -> List[Tool]:
        """제공할 도구 목록"""
        return [
            Tool(
                name="tool_name",
                description="도구 설명",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "number"}
                    },
                    "required": ["param1"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict) -> List[TextContent]:
        """도구 실행"""
        if name == "tool_name":
            # 비즈니스 로직
            result = await self._do_something(arguments)
            return [TextContent(type="text", text=json.dumps(result))]

# Step 2: FastAPI로 MCP 서버 배포
from fastapi import FastAPI
import uvicorn

app = FastAPI()
mcp_server = YourCustomMCP()

@app.post("/mcp/list-tools")
async def list_tools():
    return await mcp_server.list_tools()

@app.post("/mcp/call-tool")
async def call_tool(request: dict):
    return await mcp_server.call_tool(
        request['name'],
        request['arguments']
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. AI Agent 구현

```python
# Agent가 MCP 서버들을 오케스트레이션
from anthropic import Anthropic
import httpx

class AIAgent:
    def __init__(self):
        self.client = Anthropic(api_key="your-api-key")
        self.mcp_servers = {
            "database": "http://database-mcp:8000",
            "email": "http://email-mcp:8001",
            "ocr": "http://ocr-mcp:8002",
            # ... 더 많은 MCP 서버들
        }

    async def execute_task(self, user_request: str):
        """사용자 요청을 처리하는 메인 루프"""

        # 1. LLM에게 작업 계획 수립 요청
        system_prompt = f"""
        당신은 AI Agent입니다. 사용할 수 있는 MCP 서버들:

        {self._list_available_tools()}

        사용자 요청을 분석하고, 어떤 MCP 도구들을 어떤 순서로 호출해야 하는지 계획하세요.
        """

        # 2. LLM과 대화하면서 MCP 도구 호출
        messages = [{"role": "user", "content": user_request}]

        while True:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=system_prompt,
                messages=messages,
                tools=self._get_all_tools()
            )

            # 3. LLM이 도구 사용을 요청하면 실행
            if response.stop_reason == "tool_use":
                for content in response.content:
                    if content.type == "tool_use":
                        # MCP 서버 호출
                        result = await self._call_mcp_tool(
                            content.name,
                            content.input
                        )

                        # 결과를 LLM에게 반환
                        messages.append({
                            "role": "assistant",
                            "content": response.content
                        })
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result
                            }]
                        })
            else:
                # 최종 응답
                return response.content[0].text

    async def _call_mcp_tool(self, tool_name: str, arguments: dict):
        """MCP 서버의 도구 호출"""
        # 도구 이름에서 서버 식별 (예: "database-mcp.query" → database-mcp)
        server_name, method = tool_name.split('.')

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.mcp_servers[server_name]}/mcp/call-tool",
                json={"name": method, "arguments": arguments}
            )
            return response.text()
```

### 3. Docker Compose로 전체 시스템 배포

```yaml
# docker-compose.yml
version: '3.8'

services:
  # AI Agent
  agent:
    build: ./agent
    ports:
      - "8080:8080"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - database-mcp
      - email-mcp
      - ocr-mcp

  # MCP 서버들
  database-mcp:
    build: ./mcp-servers/database
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=postgres
      - DB_PASSWORD=${DB_PASSWORD}

  email-mcp:
    build: ./mcp-servers/email
    ports:
      - "8001:8001"
    environment:
      - SMTP_HOST=${SMTP_HOST}
      - SMTP_PASSWORD=${SMTP_PASSWORD}

  ocr-mcp:
    build: ./mcp-servers/ocr
    ports:
      - "8002:8002"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # 데이터베이스
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

### 4. 보안 고려사항

```python
# 1. API 키 보호
from cryptography.fernet import Fernet
import os

class SecureConfig:
    """민감 정보 암호화"""
    def __init__(self):
        self.cipher = Fernet(os.getenv("ENCRYPTION_KEY").encode())

    def encrypt(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt(self, encrypted: str) -> str:
        return self.cipher.decrypt(encrypted.encode()).decode()

# 2. Rate Limiting
from fastapi import HTTPException
from datetime import datetime, timedelta

class RateLimiter:
    """MCP 서버 호출 제한"""
    def __init__(self, max_calls: int = 100, window: int = 60):
        self.max_calls = max_calls
        self.window = timedelta(seconds=window)
        self.calls = {}

    def check(self, client_id: str) -> bool:
        now = datetime.now()
        if client_id not in self.calls:
            self.calls[client_id] = []

        # 오래된 기록 제거
        self.calls[client_id] = [
            call_time for call_time in self.calls[client_id]
            if now - call_time < self.window
        ]

        if len(self.calls[client_id]) >= self.max_calls:
            raise HTTPException(status_code=429, detail="Too many requests")

        self.calls[client_id].append(now)
        return True

# 3. Input Validation
from pydantic import BaseModel, validator

class MCPRequest(BaseModel):
    """MCP 요청 검증"""
    tool_name: str
    arguments: dict

    @validator('tool_name')
    def validate_tool_name(cls, v):
        # SQL Injection, Command Injection 방지
        if any(char in v for char in [';', '--', '/*', '*/', '|', '&']):
            raise ValueError("Invalid tool name")
        return v

# 4. Audit Logging
import logging

class AuditLogger:
    """모든 MCP 호출 기록"""
    def __init__(self):
        self.logger = logging.getLogger("audit")

    def log_call(self, user_id: str, mcp_name: str, tool_name: str, arguments: dict):
        self.logger.info({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "mcp": mcp_name,
            "tool": tool_name,
            "arguments": self._sanitize(arguments)
        })

    def _sanitize(self, data: dict) -> dict:
        """민감 정보 마스킹"""
        sensitive_keys = ['password', 'api_key', 'token', 'secret']
        return {
            k: "***REDACTED***" if k.lower() in sensitive_keys else v
            for k, v in data.items()
        }
```

---

## 📚 학습 로드맵

### 1단계: MCP 기초 (1-2주)
- [ ] MCP 프로토콜 이해
- [ ] 간단한 MCP 서버 구현 (Calculator MCP)
- [ ] FastAPI로 배포

### 2단계: LLM Agent 개발 (2-3주)
- [ ] Claude/OpenAI API 통합
- [ ] Tool Calling 메커니즘 이해
- [ ] Agent 루프 구현

### 3단계: 실전 프로젝트 (4-6주)
- [ ] 산업별 MCP 서버 개발
- [ ] OCR/Vision AI MCP 구현
- [ ] 프로덕션 배포 (Docker, K8s)
- [ ] 모니터링 & 로깅

### 4단계: 고급 주제 (지속적)
- [ ] Multi-Agent 협업
- [ ] Streaming 응답
- [ ] 비용 최적화
- [ ] 성능 튜닝

---

## 🎓 참고 자료

- **MCP 공식 문서**: https://modelcontextprotocol.io
- **Anthropic Claude API**: https://docs.anthropic.com
- **OpenAI Assistants API**: https://platform.openai.com/docs/assistants
- **LangChain**: https://docs.langchain.com (Agent 프레임워크)
- **FastAPI**: https://fastapi.tiangolo.com
- **Docker & Kubernetes**: 컨테이너 배포

---

**이 가이드 완성!** 🎉

이제 여러분은 20개 이상의 산업 분야에서 MCP + AI Agent 아키텍처를 구현할 수 있는 완전한 지식을 갖추었습니다. OCR, Vision AI, LLM을 결합하여 실무 문제를 해결하는 온프레미스 AI 시스템을 구축할 수 있습니다.

**핵심 메시지:**
- MCP는 AI의 HTTP다
- Agent는 여러 MCP를 오케스트레이션한다
- OCR/Vision AI로 문서/이미지 자동 처리
- 마이크로서비스처럼 독립적으로 배포
- 95-99% 시간 절감, 60-90% 비용 절감 가능

**다음 단계:**
1. 간단한 MCP 서버부터 시작
2. 자신의 산업/도메인에 적용
3. 점진적으로 확장
4. 프로덕션 배포
5. ROI 측정 및 최적화

행운을 빕니다! 🚀
