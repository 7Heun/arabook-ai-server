"""텍스트 생성 API (프롬프트 → OpenAI)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import provide_llm
from app.schemas import GenerateMessageRequest, GenerateResponse, ProblemResponse
from app.services.llm import LLMService

router = APIRouter(tags=["generate"])


@router.post(
    "/generate",
    response_model=GenerateResponse,
    responses={502: {"model": ProblemResponse, "description": "OpenAI 호출 실패"}},
    summary="프롬프트로 텍스트 생성",
)
async def generate_text(
    request: GenerateMessageRequest,
    llm: LLMService = Depends(provide_llm),
) -> GenerateResponse:
    generated = await llm.generate(
        prompt=request.prompt, model=request.model, max_tokens=request.max_tokens
    )
    return GenerateResponse(generated_text=generated)
