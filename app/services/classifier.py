"""LLM으로 도서 카테고리(대분류 1개 + 소분류 여러 개)를 분류한다.

프롬프트와 파싱 규칙은 기존 `category_classifier.py`를 그대로 옮겼다. 기존
`ChatOpenAI(model_name="gpt-4")`는 temperature 기본값(0.7)을 쓰므로, 동작을
바꾸지 않기 위해 여기서도 temperature를 지정하지 않는다.

바뀐 점
- 모듈 import 시점에 클라이언트를 만들지 않는다(키 없이도 import 가능).
- 분류 실패를 `(None, [])`로 돌려주지 않고 예외로 올린다. 그래야 `/add-book`이
  None을 DB에 넣으려다 500 나는 대신 502로 원인을 알려준다.
"""

from __future__ import annotations

import logging

from app.core.errors import LanguageModelError
from app.services.llm import LLMService

logger = logging.getLogger("arabook.classifier")

BASE_PROMPT = """웹사이트에서 이 책의 정보를 검색해서 카테고리를 분류해줘.
    분류될 수 있는 카테고리의 대분류는 다음과 같아: 소설, 시/에세이, 자기계발, 인문, 여행, 철학, 사회, 과학, 역사, 판타지/무협지.
    각 대분류에 따른 소분류는 다음과 같아:
    소설 - 추리/스릴러, SF, 판타지, 공포, 영화/드라마 원작, 역사, 사랑, 청소년,
    시/에세이 - 시, 일상, 위로, 직업, 여행, 사랑/가족, 음식,
    자기계발 - 말하기/협상/프레젠테이션, 시간관리, 습관, 글쓰기, 독서, 사고법, 리더십, 직장인, 기획, 자존감/가치관,
    인문 - 인문학, 문명, 문화, 심리학, 인간/인류, 신화, 언어, 사랑, 영화, 예체능,
    여행 - 한국, 일본, 중국, 대만/홍콩, 미국, 호주, 남미, 동남아, 유럽, 중동/아프리카, 기타 국가,
    철학 - 동양, 서양, 예술/문화, 정치/경제,
    사회 - 사회학, 정치, 법, 젠더/페미니즘, 노동, 국가, 교육, 범죄, 환경, 세계, 사회문제, 미디어, 시사 매거진,
    과학 - 물리학/공학, 수학, 화학, 천문학/지구과학, 생명과학, 인체/뇌, 과학 매거진,
    역사 - 한국 고대사, 조선사, 한국 근현대사, 세계사,
    판타지/무협지 - 무협, 퓨전 판타지, 현대 판타지, 해외 판타지, 게임 판타지, 전쟁/대체역사, 라이트노벨.
    카테고리의 대분류는 하나만 가질 수 있고, 소분류는 여러 개 가질 수 있어.
    책의 카테고리를 분류해줘. 
    형식은 '대분류 - 소분류1, 소분류2, 소분류3'으로 해줘.

    예시:
    대분류 - 소분류1, 소분류2, 소분류3
    반드시 예시 형식에 맞춰 응답해줘.

    책 정보:
    - 제목: {title}
    - 저자: {author}
    - ISBN: {isbn}
    - 설명: {description}
    - 해시태그: {hashtags}
    """


class CategoryClassifier:
    def __init__(self, llm: LLMService, model: str = "gpt-4"):
        self._llm = llm
        self._model = model

    @staticmethod
    def build_prompt(
        title: str, author: str, isbn: str, description: str, hashtags: list[str]
    ) -> str:
        return BASE_PROMPT.format(
            title=title, author=author, isbn=isbn, description=description, hashtags=hashtags
        )

    async def classify(
        self,
        title: str,
        author: str,
        isbn: str,
        description: str,
        hashtags: list[str],
    ) -> tuple[str, list[str]]:
        prompt = self.build_prompt(title, author, isbn, description, hashtags)
        response_text = (await self._llm.generate(prompt=prompt, model=self._model)).strip()
        if not response_text:
            raise LanguageModelError("카테고리 분류 응답이 비어 있습니다.")
        try:
            return self.parse(response_text)
        except (IndexError, ValueError) as exc:
            logger.warning("카테고리 응답 형식이 예상과 다릅니다: %r", response_text[:200])
            raise LanguageModelError(f"카테고리 응답을 해석하지 못했습니다: {exc}") from exc

    @staticmethod
    def parse(response_text: str) -> tuple[str, list[str]]:
        """'대분류 - 소분류1, 소분류2' 형식을 파싱한다.

        기존 로직을 그대로 유지하되, 대분류에 쉼표가 섞여 들어오는 경우(모델이
        '소설, 역사'처럼 두 개를 주는 경우)를 막기 위해 **첫 번째 대분류만**
        남긴다. 저장된 카탈로그에 실제로 이런 값이 3권 있었다.
        """
        text = response_text.strip()
        if "대분류" in text:
            lines = [line for line in text.splitlines() if line.strip()]
            if len(lines) < 2:
                raise ValueError("대분류/소분류 두 줄이 필요합니다.")
            main_part = lines[0].split(" -", 1)[1]
            sub_part = lines[1].split(" -", 1)[1]
        else:
            main_part, sub_part = text.split(" -", 1)

        main_category = main_part.strip().split(",")[0].strip()
        sub_categories = [part.strip() for part in sub_part.split(",") if part.strip()]
        if not main_category:
            raise ValueError("대분류가 비어 있습니다.")
        if not sub_categories:
            raise ValueError("소분류가 비어 있습니다.")
        return main_category, sub_categories
