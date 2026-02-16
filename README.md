# JFBench

JFBench — CLI-инструмент для бенчмаркинга LLM-моделей на задаче извлечения и приведения сырого текста к целевому JSON-формату по заранее заданным схемам и кейсам.

Проект ориентирован на воспроизводимые прогоны:
- данные кейсов и схем лежат в файловой структуре;
- конфигурация прогона задается через `csv/xlsx`;
- результаты сохраняются в отчет (`csv` или `xlsx`) с метриками качества.

## Зачем нужен проект

JFBench помогает сравнивать модели в одинаковых условиях:
- одна и та же входная выборка (`cases`);
- одна и та же JSON Schema (`schemas`);
- одинаковые системные промпты (`prompts`);
- измеримые метрики похожести (`similarity`, `field_match`, `value_match`).

Это удобно для:
- выбора модели под конкретный формат данных;
- регрессионного контроля качества при смене модели;
- документирования результатов экспериментов.

## Стек и требования

- Python `>=3.13`
- пакетный менеджер и раннер: `uv`
- CLI: `click`
- валидация JSON Schema: `jsonschema`
- табличные данные и экспорт: `polars`
- тесты: `pytest`
- линт/форматирование: `ruff`

## Быстрый старт

### 1. Клонировать и установить зависимости

```bash
git clone <repo-url>
cd jfb
uv sync --dev
```

### 2. Проверить CLI

```bash
uv run jfb --help
```

После установки в `pyproject.toml` зарегистрирован скрипт:
- `jfb = "models.cli:main"`

## Основные сценарии CLI

### 1. Инициализация рабочей директории кейсов

Создает структуру каталогов (`cases`, `schemas`, `runs`, `prompts`):

```bash
uv run jfb --new /absolute/path/to/benchmark
```

### 2. Запуск бенчмарка

```bash
uv run jfb PATH PROVIDER RUN_CONFIG OUTPUT_PATH [OPTIONS]
```

Где:
- `PATH` — путь к корневой директории кейсов.
- `PROVIDER` — провайдер из enum приложения (сейчас: `lmstudio`).
- `RUN_CONFIG` — путь к `csv/xlsx` конфигу прогона.
  - если передан относительный путь и файл не найден в текущей директории, он ищется в `PATH/runs`.
  - имя файла без расширения используется как `run_name` в отчете.
- `OUTPUT_PATH` — файл результата (`.csv` или `.xlsx`).

Опции:
- `--api-host` — адрес провайдера в формате `host:port` (по умолчанию `localhost:1234`).
- `-v`, `-vv` — уровень подробности логов.
- `--quiet` — только ошибки.
- `--log-level` — явный уровень (`DEBUG|INFO|WARNING|ERROR|CRITICAL`).

Пример:

```bash
uv run jfb /absolute/path/to/benchmark lmstudio demo.csv /absolute/path/to/results.csv --api-host localhost:1234 -v
```

## Формат данных

### Структура директории

```text
benchmark/
  cases/
  schemas/
  prompts/
  runs/
```

### `cases/<case_name>.json`

```json
{
  "raw": "Сырой текст для обработки моделью",
  "expected_value": {
    "answer": "ok"
  },
  "schema": "answer.schema.json"
}
```

Поля:
- `raw` — исходный текст для модели;
- `expected_value` — эталонный JSON;
- `schema` — имя файла схемы (из `schemas/`).

### `schemas/<schema_name>.json`

Обычная JSON Schema (draft 2020-12), которая:
- передается в модель как `response_format`;
- используется для валидации ответа модели.

### `prompts/<case_name>.txt`

Системный промпт для конкретного кейса.

### `runs/<run_config>.csv|xlsx`

Обязательные колонки:
- `model_id`
- `case_name`

Пример `csv`:

```csv
model_id,case_name
model-a,invoice_case
model-b,invoice_case
```

## Что происходит во время прогона

Для каждой строки `run_config`:
1. Загружается кейс (`cases/<case_name>.json`).
2. Загружается и компилируется схема (с кешированием).
3. Загружается системный промпт (с кешированием).
4. Выполняется запрос к LLM-провайдеру.
5. Ответ валидируется по JSON Schema.
6. Считаются метрики похожести относительно `expected_value`.
7. Формируется строка отчета.

На выходе формируется таблица с полями:
- идентификатор/имя прогона;
- кейс, модель, схема;
- `similarity`, `field_match`, `value_match` и счетчики;
- время ответа;
- эталон, фактический результат и текст ошибки (если была).

## Разработка для контрибьюторов

### Рекомендованный цикл разработки

1. Создать ветку:
```bash
git checkout -b codex/<short-feature-name>
```

2. После правок прогнать форматтер и линтер:
```bash
uv run ruff format <changed_paths>
uv run ruff check <changed_paths>
```

3. Прогнать целевые тесты, затем полный набор:
```bash
uv run pytest -q tests/<target_test>.py
uv run pytest -q tests
```

4. Если затрагивалась типизация клиентов:
```bash
uv run pyright tests/test_llm_clients.py
```

### Pre-commit (рекомендуется)

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### Где что находится

- `src/models/case.py` — управление директориями кейсов, загрузка `case` и `run_config`.
- `src/models/llm_clients/` — адаптеры провайдеров (сейчас LMStudio).
- `src/models/repositories/` — кеши схем/промптов и LRU-кеш.
- `src/models/estimator.py` — вычисление метрик похожести.
- `src/models/report.py` — генерация отчета.
- `src/models/cli.py` — orchestration CLI-пайплайна.
- `tests/` — unit-тесты.

## Ограничения и важные замечания

- На текущем этапе поддерживается провайдер `lmstudio`.
- `--api-host` должен быть строго в формате `host:port` (без `http://`).
- Для чтения `xlsx` run-config может потребоваться дополнительная зависимость `fastexcel` (ограничение `polars.read_excel`).
- `OUTPUT_PATH` поддерживает только `.csv` и `.xlsx`.

## Лицензия

Пока не определена. Добавьте файл `LICENSE` при необходимости.
