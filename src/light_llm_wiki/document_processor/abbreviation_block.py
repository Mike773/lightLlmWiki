def format_for_prompt(items: list[tuple[str, str | None]]) -> str:
    if not items:
        return "(в направлении аббревиатур не обнаружено)"
    lines = []
    for name, description in items:
        if description is None:
            lines.append(f"- {name} — расшифровка не найдена в документе")
        else:
            lines.append(f"- {name} — {description}")
    return "\n".join(lines)


def format_exclude_list(items: list[tuple[str, str | None]]) -> str:
    if not items:
        return "(в направлении аббревиатур не обнаружено)"
    return "\n".join(f"- {name}" for name, _ in items)
