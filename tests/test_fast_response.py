from __future__ import annotations

from app.pipeline.fast_response import build_fast_response, detect_fast_intent


def test_fast_response_handles_greeting() -> None:
    resp = build_fast_response("Xin chào", include_debug=True)

    assert resp is not None
    assert resp.structured["intent"] == "greeting"
    assert resp.confidence == "high"
    assert resp.citations == []
    assert resp.debug is not None
    assert "fast_path_ms" in resp.debug.timings_ms


def test_fast_response_handles_out_of_scope_short_question() -> None:
    resp = build_fast_response("Thời tiết hôm nay thế nào?")

    assert resp is not None
    assert resp.structured["intent"] == "out_of_scope"
    assert "pháp luật hình sự" in resp.final_answer


def test_fast_response_does_not_treat_toi_as_crime_word() -> None:
    resp = build_fast_response("Tôi muốn hỏi thời tiết hôm nay")

    assert resp is not None
    assert resp.structured["intent"] == "out_of_scope"


def test_fast_response_does_not_intercept_legal_question() -> None:
    assert detect_fast_intent("Hành vi trên thuộc tội gì?") is None
    assert build_fast_response("A cướp tài sản 100 triệu thì phạm tội gì?") is None
