import re

def extract_sql_from_text(text: str) -> str:
    """
    LLM 출력에서 SQL문만 추출
    - 입력 구조 무관
    - SELECT ~ FROM/WHERE/GROUP BY/... 기준
    - 줄바꿈, 탭 제거
    - 연속 스페이스 2개 이상 → 1개
    """
    # 1. SQL문 시작: SELECT 키워드
    sql_start = re.search(r'\bSELECT\b', text, re.IGNORECASE)
    if not sql_start:
        return ""

    # 2. SQL문 끝 추정: 세미콜론이나 다음 ###/JSON/CSV 같은 패턴
    #    세미콜론 없으면 다음 비SQL 키워드까지
    end_patterns = [r';', r'###', r'\[', r'CountryId,', r'\{']
    sql_end = len(text)
    for pattern in end_patterns:
        match = re.search(pattern, text[sql_start.start():])
        if match:
            sql_end = sql_start.start() + match.start()
            break

    sql_text = text[sql_start.start():sql_end]

    # 3. 줄바꿈, 탭, 캐리지 리턴 제거
    sql_text = re.sub(r'[\n\t\r]', '', sql_text)

    # 4. 연속 스페이스 2개 이상 → 1개로
    sql_text = re.sub(r' {2,}', ' ', sql_text)

    # 5. 앞뒤 공백 제거
    sql_text = sql_text.strip()

    return sql_text
