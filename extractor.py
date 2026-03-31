import re

QUESTION_PATTERN = re.compile(r"^\s*(\d{1,3})\s*[\.\)]\s*(.+)$")
OPTION_PATTERN = re.compile(r"^\s*([a-dA-D])\s*[\)\.\]:-]\s*(.+)$")
LOOSE_OPTION_PATTERN = re.compile(r"^\s*([A-Za-z0-9])\s*[\)\.\]:-]\s*(.+)$")
INLINE_MARKER_PATTERN = re.compile(r"(?<!\S)([a-dA-DoO0lI1-48])\s*[\)\.\]:-]\s*")


def _normalize_ocr_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    if not line:
        return ""

    # Common OCR confusions for option markers at line start.
    # e.g., "I) Paris" -> "a) Paris", "2) Delhi" -> "b) Delhi", "o) foo" -> "a) foo"
    line = re.sub(r"^[oO0]\s*[\)\.\]:-]\s*", "a) ", line)
    line = re.sub(r"^[lI1]\s*[\)\.\]:-]\s*", "a) ", line)
    line = re.sub(r"^[2zZ]\s*[\)\.\]:-]\s*", "b) ", line)
    line = re.sub(r"^[3]\s*[\)\.\]:-]\s*", "c) ", line)
    line = re.sub(r"^[4]\s*[\)\.\]:-]\s*", "d) ", line)
    return line


def _normalize_option_key(raw_key: str):
    key = raw_key.lower()
    mapping = {
        "a": "a",
        "b": "b",
        "c": "c",
        "d": "d",
        "o": "a",
        "0": "a",
        "l": "a",
        "i": "a",
        "1": "a",
        "2": "b",
        "3": "c",
        "4": "d",
        "8": "d",
    }
    return mapping.get(key)


def _extract_inline_options(line: str):
    """
    Handle OCR where full MCQ appears in one line:
    "SQL ... a) Database b) Design c) AI d) Game"
    """
    matches = list(INLINE_MARKER_PATTERN.finditer(line))
    if len(matches) < 1:
        return None

    first_marker_start = matches[0].start(1)
    question = line[:first_marker_start].strip(" :-")
    if not question:
        question = "Question text not confidently recognized"

    options = {}
    ordered_keys = ["a", "b", "c", "d"]
    next_idx = 0

    for i, m in enumerate(matches):
        key = _normalize_option_key(m.group(1))
        value_start = m.end()
        value_end = matches[i + 1].start(1) if i + 1 < len(matches) else len(line)
        value = line[value_start:value_end].strip(" :-")
        if not value:
            continue

        if key is None or key in options:
            if next_idx >= len(ordered_keys):
                continue
            key = ordered_keys[next_idx]
        options[key] = value
        if next_idx < len(ordered_keys) and key == ordered_keys[next_idx]:
            next_idx += 1

    # If first detected marker was not "a", often OCR merged option A into question tail.
    # Recover last token(s) from question as option A.
    first_key = _normalize_option_key(matches[0].group(1))
    if first_key and first_key != "a" and "a" not in options:
        q_tokens = question.split()
        if q_tokens:
            recovered_a = q_tokens[-1]
            options = {"a": recovered_a, **options}
            question = " ".join(q_tokens[:-1]).strip() or question

    if len(options) >= 2:
        return {"question": question, "options": options}
    return None


def extract_mcqs(text):
    lines = [_normalize_ocr_line(line) for line in text.split("\n")]
    lines = [ln for ln in lines if ln]

    questions = []
    current_q = None
    pending_question_lines = []
    expected_option_order = ["a", "b", "c", "d"]

    for line in lines:
        if not line:
            continue

        inline_mcq = _extract_inline_options(line)
        if inline_mcq:
            if current_q:
                questions.append(current_q)
                current_q = None
            questions.append(inline_mcq)
            pending_question_lines = []
            continue

        q_match = QUESTION_PATTERN.match(line)
        o_match = OPTION_PATTERN.match(line)
        loose_o_match = LOOSE_OPTION_PATTERN.match(line)

        if q_match:
            if current_q:
                questions.append(current_q)

            current_q = {
                "question": q_match.group(2).strip(),
                "options": {}
            }
            pending_question_lines = []

        elif o_match and current_q:
            key = o_match.group(1).lower()
            current_q["options"][key] = o_match.group(2).strip()

        elif o_match and not current_q:
            # Handwritten OCR may miss numeric question index.
            current_q = {
                "question": " ".join(pending_question_lines).strip(),
                "options": {}
            }
            pending_question_lines = []
            key = o_match.group(1).lower()
            current_q["options"][key] = o_match.group(2).strip()

        elif loose_o_match:
            # Fallback: recover option sequence even when marker is misread.
            value = loose_o_match.group(2).strip()
            if not current_q:
                current_q = {
                    "question": " ".join(pending_question_lines).strip(),
                    "options": {}
                }
                pending_question_lines = []
            existing_keys = list(current_q["options"].keys())
            next_idx = min(len(existing_keys), len(expected_option_order) - 1)
            guessed_key = expected_option_order[next_idx]
            if guessed_key not in current_q["options"]:
                current_q["options"][guessed_key] = value
            else:
                current_q["options"][existing_keys[-1]] += " " + value

        else:
            if current_q and current_q["options"]:
                # Continuation line: append to latest option if available else question.
                last_key = list(current_q["options"].keys())[-1]
                current_q["options"][last_key] += " " + line
            elif current_q and not current_q["options"]:
                current_q["question"] = f'{current_q["question"]} {line}'.strip()
            else:
                pending_question_lines.append(line)

    if current_q:
        questions.append(current_q)

    # If we got options but empty question, try infer question from text before first option marker.
    for q in questions:
        if not q.get("question", "").strip():
            q["question"] = "Question text not confidently recognized"

    # Keep only likely complete MCQs.
    filtered = []
    for q in questions:
        opt_count = len(q.get("options", {}))
        if q.get("question", "").strip() and opt_count >= 2:
            filtered.append(q)

    # Final fallback for handwritten OCR noise:
    # if nothing parsed but we have enough lines, treat last 4 text lines as options.
    if not filtered:
        usable = [
            ln for ln in lines
            if not re.fullmatch(r"[\d\W_]+", ln) and len(ln.strip()) >= 2
        ]
        if len(usable) >= 5:
            question_text = " ".join(usable[:-4]).strip()
            options = usable[-4:]
            if question_text:
                filtered.append(
                    {
                        "question": question_text,
                        "options": {
                            "a": options[0],
                            "b": options[1],
                            "c": options[2],
                            "d": options[3],
                        },
                    }
                )

    return filtered