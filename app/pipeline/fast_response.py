"""Fast path cho cau hoi khong can chay RAG.

Module nay xu ly cac cau rat ngan nhu loi chao/cam on/tam biet va cac cau hoi
ngoai pham vi phap luat hinh su. Muc tieu la tra loi ngay, tranh goi NER,
Neo4j, reranker va LLM khi chac chan khong can retrieval.
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass

from app.models.schemas import ChatResponse, ChatResponseDebug


@dataclass(frozen=True)
class FastIntent:
    kind: str
    answer: str


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s?]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_count(norm: str) -> int:
    return len([part for part in norm.split(" ") if part])


_LEGAL_TERMS = (
    "blhs",
    "bo luat hinh su",
    "luat hinh su",
    "phap luat",
    "toi gi",
    "toi danh",
    "pham toi",
    "cau thanh toi",
    "hinh phat",
    "khung hinh phat",
    "muc an",
    "di tu",
    "tu hinh",
    "chung than",
    "an treo",
    "dieu",
    "khoan",
    "dong pham",
    "chu muu",
    "giup suc",
    "xui giuc",
    "truy cuu",
    "trach nhiem hinh su",
    "hanh vi",
    "bi hai",
    "nan nhan",
    "giet",
    "cuop",
    "trom",
    "lua dao",
    "ma tuy",
    "danh bac",
    "bao hanh",
    "co y gay thuong tich",
    "vo y lam chet",
    "tham o",
    "nhan hoi lo",
)

_GREETING_PATTERNS = (
    r"^(hi|hello|hey|alo|xin chao|chao|chao ban|chao anh|chao chi|chao em|ban oi|ad oi)$",
    r"^(buoi sang|buoi chieu|buoi toi) vui ve$",
)

_THANKS_PATTERNS = (
    r"^(cam on|cam on ban|thank|thanks|thank you|ok cam on|da hieu|hieu roi)$",
)

_GOODBYE_PATTERNS = (
    r"^(tam biet|bye|goodbye|hen gap lai|chao tam biet)$",
)

_ABOUT_PATTERNS = (
    r"^(ban la ai|ban lam duoc gi|ban co the lam gi|day la chatbot gi|chatbot nay lam gi)\??$",
)


def _matches_any(norm: str, patterns: tuple[str, ...]) -> bool:
    return any(re.match(pattern, norm) for pattern in patterns)


def _has_legal_signal(norm: str) -> bool:
    return any(term in norm for term in _LEGAL_TERMS)


def detect_fast_intent(question: str) -> FastIntent | None:
    """Nhan dien cac cau co the tra loi nhanh ma khong can RAG."""
    norm = _normalize(question)
    if not norm:
        return FastIntent(
            kind="empty",
            answer="Bạn hãy nhập câu hỏi về Bộ luật Hình sự để mình hỗ trợ nhé.",
        )

    if _matches_any(norm, _GREETING_PATTERNS) and _word_count(norm) <= 5:
        return FastIntent(
            kind="greeting",
            answer=(
                "Chào bạn, mình là chatbot hỗ trợ tra cứu và phân tích tình huống "
                "theo Bộ luật Hình sự. Bạn có thể mô tả vụ việc hoặc hỏi về tội danh, "
                "điều luật, khung hình phạt."
            ),
        )

    if _matches_any(norm, _THANKS_PATTERNS) and _word_count(norm) <= 5:
        return FastIntent(
            kind="thanks",
            answer="Rất vui được hỗ trợ bạn. Khi cần phân tích tình huống hình sự, bạn cứ gửi nội dung vụ việc nhé.",
        )

    if _matches_any(norm, _GOODBYE_PATTERNS) and _word_count(norm) <= 5:
        return FastIntent(
            kind="goodbye",
            answer="Tạm biệt bạn. Khi cần tra cứu Bộ luật Hình sự, bạn quay lại hỏi mình nhé.",
        )

    if _matches_any(norm, _ABOUT_PATTERNS):
        return FastIntent(
            kind="about",
            answer=(
                "Mình là chatbot RAG hỗ trợ hỏi đáp về Bộ luật Hình sự: nhận diện tình huống, "
                "gợi ý tội danh, điều/khoản liên quan, khung hình phạt và trích dẫn căn cứ."
            ),
        )

    if _has_legal_signal(norm):
        return None

    if _word_count(norm) <= 25:
        return FastIntent(
            kind="out_of_scope",
            answer=(
                "Mình hiện chỉ hỗ trợ nội dung liên quan đến pháp luật hình sự. "
                "Bạn hãy đặt câu hỏi về tội danh, điều luật, khung hình phạt hoặc mô tả một tình huống vụ việc nhé."
            ),
        )

    return None


def build_fast_response(question: str, include_debug: bool = False) -> ChatResponse | None:
    """Tra ChatResponse nhanh neu cau hoi thuoc fast path."""
    t0 = time.time()
    intent = detect_fast_intent(question)
    if intent is None:
        return None

    debug = None
    if include_debug:
        debug = ChatResponseDebug()
        debug.timings_ms["fast_path_ms"] = round((time.time() - t0) * 1000, 1)
        debug.warnings.append(f"fast_path:{intent.kind}")

    return ChatResponse(
        question=question,
        final_answer=intent.answer,
        structured={
            "type": "fast_response",
            "intent": intent.kind,
            "handled_by": "rule_based_fast_path",
        },
        citations=[],
        confidence="high",
        debug=debug,
    )
