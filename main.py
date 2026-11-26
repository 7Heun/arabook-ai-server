"""하위 호환 진입점.

예전에는 `uvicorn main:app`으로 실행했다. 구조를 `app/` 패키지로 옮긴 뒤에도
같은 명령이 동작하도록 이 파일만 남겨 둔다. 새 코드는 `app.main:app`을 쓴다.
"""

from app.main import app

__all__ = ["app"]
