"""Normalize pytest and Python tracebacks without retaining paths or input values."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorSignature:
    exception_type: str
    message_template: str
    library_symbol: str = ""

    def __str__(self) -> str:
        return f"{self.exception_type}: {self.message_template}" + (
            f" [{self.library_symbol}]" if self.library_symbol else ""
        )


def _error_spec():
    from .targets import active

    return active().spec.errors


class ErrorParser:
    """库专用的归一化规则(如把校验错误码合成一个签名)来自目标包 `errors.signature_rules`。"""

    def parse(self, text: str, message: str = "", kind: str = "failure") -> ErrorSignature:
        for rule in _error_spec().signature_rules:
            codes = sorted(set(re.findall(rule.pattern, text + "\n" + message)))
            if codes:
                return ErrorSignature(rule.exception, rule.join.join(codes), rule.symbol)
        lines = [re.sub(r"^E\s+", "", ln).strip() for ln in text.splitlines()]
        exceptions = [ln for ln in lines if re.match(r"^(?:[\w.]+\.)?[\w]+(?:Error|Exception|Warning):", ln)]
        e_lines = [re.sub(r"^E\s+", "", ln).strip() for ln in text.splitlines() if re.match(r"^E\s+", ln)]
        base = (
            exceptions[-1]
            if exceptions
            else (
                e_lines[0]
                if e_lines
                else (message.splitlines()[0] if message else (lines[-1] if lines else kind))
            )
        )
        exception, sep, body = base.partition(":")
        if not sep or not re.fullmatch(r"[\w.]+(?:Error|Exception|Warning)", exception):
            exception, body = kind, base
        exception = exception.rsplit(".", 1)[-1]
        symbols = [
            s for s in re.findall(r"`([^`]+)`", body) if re.fullmatch(r"@?[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", s)
        ]
        symbol = symbols[0] if symbols else ""
        body = re.sub(r"https?://\S+", "<url>", body)
        body = re.sub(r"0x[0-9a-fA-F]+", "0x?", body)
        body = re.sub(r"(?:[A-Za-z]:\\|/)[^\s:]+:\d+", "<path>:<n>", body)
        if exception != kind:
            body = re.sub(r"\b\d+\b", "<n>", body)
        body = re.sub(r"\s+", " ", body).strip()
        return ErrorSignature(exception, body[:300], symbol)
