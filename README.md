# light-llm-wiki

База знаний для агента-аналитика на Python. Помогает отвечать на вопросы по
направлениям (`direction`) и анализировать метрики.

## Принципы

- **Изоляция по direction.** Загрузка и получение данных всегда выполняются в
  рамках одного направления — direction является основной осью разграничения.
- **PostgreSQL + pgvector.** Хранилище — PostgreSQL с расширением `vector`.
- **Одна миграция-файл.** В `migrations/schema.sql` лежит актуальная версия
  схемы целиком (без последовательности версий). Индексы добавляются вручную.
- **Простота.** Минимум таблиц и сущностей.

## Структура

```
migrations/
  schema.sql              # актуальная версия схемы БД
src/light_llm_wiki/
  document_processor/     # пайплайн индексации документов
  query/                  # ответ на вопрос по базе знаний
  db.py                   # доступ к PostgreSQL
  llm.py / embedding.py   # OpenAI-совместимые клиенты
pyproject.toml
```

## Применение миграции

```bash
psql "$DB_URL" -f migrations/schema.sql
```

## Установка

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

После установки доступны команды `light-llm-wiki-pipeline` (индексация) и
`light-llm-wiki-query` (запрос к базе).

## Конфигурация

Через переменные окружения:

| Переменная | Назначение | Дефолт |
|---|---|---|
| `OPENAI_API_KEY` | ключ OpenAI или совместимого провайдера | (обязательно) |
| `DSN` | строка подключения к PostgreSQL | `postgresql://postgres:postgres@localhost:5432/light_llm_wiki` |
| `CHAT_MODEL` | chat-модель для извлечения и ответа | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | модель эмбеддингов | `text-embedding-3-large` |
| `EMBEDDING_DIMENSIONS` | размерность вектора (должна совпадать с `vector(N)` в схеме) | `2560` |
| `OPENAI_BASE_URL` | endpoint для OpenAI-совместимых сервингов (TEI, vLLM, Ollama) | (none) |

Любое значение можно переопределить флагом командной строки —
см. `--help` у каждой утилиты.

## Загрузка документа

Документ кладётся прямо в `llm_wiki_rag.documents`. Direction обязан
существовать заранее.

```bash
psql "$DSN" -c "
INSERT INTO llm_wiki_rag.directions (key, name, description)
VALUES ('analytics', 'Аналитика', 'Описание направления')
ON CONFLICT (key) DO NOTHING;
"

psql "$DSN" <<'SQL'
INSERT INTO llm_wiki_rag.documents (direction_key, title, content)
VALUES ('analytics', 'Название документа', $$
Текст документа целиком, многострочно.
$$)
RETURNING id;
SQL
```

Запомни `id` из `RETURNING` — его передаёт CLI.

## Прогон пайплайна

Все стейджи последовательно:

```bash
light-llm-wiki-pipeline --document-id <ID>
```

Один стейдж:

```bash
light-llm-wiki-pipeline --document-id <ID> --stage entities
```

Стейджи (в обязательном порядке для одного документа):

| Стейдж | Что делает |
|---|---|
| `abbreviations` | список аббревиатур → расшифровки → `stage_entities (type='abbreviation')` |
| `entities` | список сущностей (без аббревиатур) → описания → embedding → `stage_entities (type='entity')` |
| `relations` | кандидаты связей по сущностям, dedup по неупорядоченной паре, описание+цитата → `stage_relations` |
| `promote` | upsert в постоянные таблицы `entities` / `entity_relations` |
| `wiki` | LLM-summary на сущность + LLM-обзор направления → `wiki_pages` трёх типов с embedding и кросс-ссылками |

Повторный прогон того же `--document-id` идемпотентен: stage перетирается, постоянные таблицы upsert'ятся по matching без дубликатов.

## Ответ на вопрос

```bash
light-llm-wiki-query --direction analytics --question "Что такое транзакция?"
```

Можно подавать через stdin:

```bash
echo "Что такое транзакция?" | light-llm-wiki-query --direction analytics
```

Команда печатает человекочитаемую trace без id и markdown-ссылок:
какие аббревиатуры и сущности были выделены из вопроса, что нашлось в
базе, какие документы упоминают эти сущности, какие цитаты были найдены,
и финальный ответ. Если что-то не нашлось — это так и говорится.
