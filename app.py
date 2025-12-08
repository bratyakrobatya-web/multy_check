import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor
import time
import pandas as pd
from io import BytesIO
from datetime import datetime

# Топ-10 моделей по MMLU-Pro (Reasoning & Knowledge)
TOP_MODELS = [
    {
        "id": "google/gemini-2.5-pro-preview-06-05",
        "name": "Gemini 3 Pro Preview",
        "provider": "Google",
        "score": "90%"
    },
    {
        "id": "anthropic/claude-opus-4",
        "name": "Claude Opus 4.5",
        "provider": "Anthropic",
        "score": "90%"
    },
    {
        "id": "anthropic/claude-sonnet-4",
        "name": "Claude 4.5 Sonnet",
        "provider": "Anthropic",
        "score": "88%"
    },
    {
        "id": "openai/gpt-4.1",
        "name": "GPT-5.1",
        "provider": "OpenAI",
        "score": "87%"
    },
    {
        "id": "x-ai/grok-3-beta",
        "name": "Grok 4",
        "provider": "xAI",
        "score": "87%"
    },
    {
        "id": "deepseek/deepseek-chat",
        "name": "DeepSeek V3.2",
        "provider": "DeepSeek",
        "score": "86%"
    },
    {
        "id": "openai/codex-mini",
        "name": "GPT-5.1 Codex",
        "provider": "OpenAI",
        "score": "86%"
    },
    {
        "id": "x-ai/grok-3-mini-beta",
        "name": "Grok 4.1 Fast",
        "provider": "xAI",
        "score": "85%"
    },
    {
        "id": "deepseek/deepseek-r1-0528",
        "name": "DeepSeek R1 0528",
        "provider": "DeepSeek",
        "score": "85%"
    },
    {
        "id": "moonshotai/kimi-k2",
        "name": "Kimi K2 Thinking",
        "provider": "Moonshot",
        "score": "85%"
    },
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def query_model(model_id: str, prompt: str, api_key: str) -> dict:
    """Запрос к одной модели через OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/multi-ai-checker",
        "X-Title": "Multi AI Checker"
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
    }

    start_time = time.time()
    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "success": True,
                "content": content,
                "time": elapsed,
                "model_id": model_id
            }
        else:
            error_msg = response.json().get("error", {}).get("message", response.text)
            return {
                "success": False,
                "error": f"Ошибка {response.status_code}: {error_msg}",
                "time": elapsed,
                "model_id": model_id
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Таймаут запроса (120 сек)",
            "time": 120,
            "model_id": model_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": time.time() - start_time,
            "model_id": model_id
        }


def create_excel(prompt: str, results: dict, selected_models: list) -> BytesIO:
    """Создает Excel файл с результатами"""
    output = BytesIO()

    # Формируем данные для Excel - каждая модель в отдельном столбце
    data = {"Запрос": [prompt]}

    for model in selected_models:
        model_id = model['id']
        col_name = f"{model['name']}\n({model['provider']})"

        if model_id in results:
            result = results[model_id]
            if result['success']:
                data[col_name] = [result['content']]
            else:
                data[col_name] = [f"ОШИБКА: {result.get('error', 'Неизвестная ошибка')}"]
        else:
            data[col_name] = ["Нет данных"]

    df = pd.DataFrame(data)

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Ответы AI')

        # Настраиваем ширину столбцов
        worksheet = writer.sheets['Ответы AI']
        worksheet.column_dimensions['A'].width = 50  # Запрос

        for idx, col in enumerate(df.columns[1:], start=2):
            col_letter = chr(64 + idx)
            worksheet.column_dimensions[col_letter].width = 60

        # Включаем перенос текста для всех ячеек
        from openpyxl.styles import Alignment
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical='top')

    output.seek(0)
    return output


def main():
    st.set_page_config(
        page_title="Multi AI Chat",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Multi AI Chat")
    st.markdown("**Один запрос — ответы от 10 лучших нейросетей → Excel**")

    # Получаем API ключ из secrets (приоритет) или из ввода
    api_key = st.secrets.get("OPENROUTER_SECRET_KEY", "")

    # Sidebar с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")

        if api_key:
            st.success("✅ API ключ загружен")
        else:
            api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                help="Получите ключ на https://openrouter.ai/keys"
            )

        st.markdown("---")
        st.subheader("📋 Выбор моделей")

        selected_models = []
        for model in TOP_MODELS:
            if st.checkbox(
                f"{model['name']} ({model['provider']}) - {model['score']}",
                value=True,
                key=f"model_{model['id']}"
            ):
                selected_models.append(model)

        st.markdown("---")
        st.markdown("""
        ### 📖 О приложении
        Ответы выгружаются в Excel с удобными столбцами.

        **API:** [OpenRouter](https://openrouter.ai)
        """)

    # Основная область
    prompt = st.text_area(
        "Введите ваш запрос:",
        height=100,
        placeholder="Например: Объясни квантовые вычисления простыми словами"
    )

    send_button = st.button("🚀 Отправить и скачать Excel", type="primary", use_container_width=True)

    if send_button:
        if not api_key:
            st.error("❌ Пожалуйста, введите API ключ OpenRouter в боковой панели")
            return

        if not prompt.strip():
            st.error("❌ Пожалуйста, введите запрос")
            return

        if not selected_models:
            st.error("❌ Пожалуйста, выберите хотя бы одну модель")
            return

        # Прогресс
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = {}

        # Параллельное выполнение запросов
        status_text.text("⏳ Отправляю запросы ко всем моделям...")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(query_model, model['id'], prompt, api_key): model
                for model in selected_models
            }

            completed = 0
            for future in futures:
                model = futures[future]
                result = future.result()
                results[model['id']] = result
                completed += 1
                progress_bar.progress(completed / len(selected_models))

                if result['success']:
                    status_text.text(f"✅ {model['name']} ответил ({completed}/{len(selected_models)})")
                else:
                    status_text.text(f"❌ {model['name']} ошибка ({completed}/{len(selected_models)})")

        progress_bar.empty()
        status_text.empty()

        # Статистика
        successful = sum(1 for r in results.values() if r['success'])
        failed = len(results) - successful

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Успешно", successful)
        col2.metric("❌ Ошибки", failed)
        col3.metric("📊 Всего", len(results))

        # Создаем Excel
        excel_file = create_excel(prompt, results, selected_models)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_responses_{timestamp}.xlsx"

        st.download_button(
            label="📥 Скачать Excel",
            data=excel_file,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )

        # Превью результатов
        st.markdown("---")
        st.subheader("👀 Превью ответов")

        for model in selected_models:
            model_id = model['id']
            if model_id in results:
                result = results[model_id]
                with st.expander(f"{model['name']} ({model['provider']})"):
                    if result['success']:
                        st.markdown(result['content'])
                    else:
                        st.error(result.get('error', 'Ошибка'))


if __name__ == "__main__":
    main()
