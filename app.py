import streamlit as st
import requests
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

# Топ-10 самых передовых текстовых моделей (декабрь 2024)
TOP_MODELS = [
    {
        "id": "anthropic/claude-sonnet-4",
        "name": "Claude Sonnet 4",
        "provider": "Anthropic"
    },
    {
        "id": "openai/gpt-4o",
        "name": "GPT-4o",
        "provider": "OpenAI"
    },
    {
        "id": "google/gemini-2.0-flash-001",
        "name": "Gemini 2.0 Flash",
        "provider": "Google"
    },
    {
        "id": "anthropic/claude-3.5-haiku",
        "name": "Claude 3.5 Haiku",
        "provider": "Anthropic"
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B",
        "provider": "Meta"
    },
    {
        "id": "deepseek/deepseek-r1",
        "name": "DeepSeek R1",
        "provider": "DeepSeek"
    },
    {
        "id": "mistralai/mistral-large-2411",
        "name": "Mistral Large",
        "provider": "Mistral AI"
    },
    {
        "id": "qwen/qwen-2.5-72b-instruct",
        "name": "Qwen 2.5 72B",
        "provider": "Alibaba"
    },
    {
        "id": "cohere/command-r-plus",
        "name": "Command R+",
        "provider": "Cohere"
    },
    {
        "id": "x-ai/grok-2-1212",
        "name": "Grok 2",
        "provider": "xAI"
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


def main():
    st.set_page_config(
        page_title="Multi AI Chat",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Multi AI Chat")
    st.markdown("**Один запрос — ответы от 10 лучших нейросетей**")

    # Sidebar с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")

        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            key="OPENROUTER_SECRET_KEY",
            help="Получите ключ на https://openrouter.ai/keys"
        )

        st.markdown("---")
        st.subheader("📋 Выбор моделей")

        selected_models = []
        for model in TOP_MODELS:
            if st.checkbox(
                f"{model['name']} ({model['provider']})",
                value=True,
                key=f"model_{model['id']}"
            ):
                selected_models.append(model)

        st.markdown("---")
        st.markdown("""
        ### 📖 О приложении
        Это приложение позволяет сравнить ответы разных AI-моделей на один и тот же вопрос.

        **API:** [OpenRouter](https://openrouter.ai)
        """)

    # Основная область
    prompt = st.text_area(
        "Введите ваш запрос:",
        height=100,
        placeholder="Например: Объясни квантовые вычисления простыми словами"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("🚀 Отправить", type="primary", use_container_width=True)
    with col2:
        parallel = st.checkbox("Параллельные запросы", value=True, help="Отправлять запросы ко всем моделям одновременно")

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

        st.markdown("---")
        st.subheader("📊 Ответы моделей")

        # Создаем контейнеры для каждой модели
        results = {}
        containers = {}

        # Создаем сетку для отображения результатов
        cols = st.columns(2)

        for idx, model in enumerate(selected_models):
            col = cols[idx % 2]
            with col:
                with st.container(border=True):
                    st.markdown(f"### {model['name']}")
                    st.caption(f"Provider: {model['provider']} | Model: `{model['id']}`")
                    containers[model['id']] = st.empty()
                    containers[model['id']].info("⏳ Ожидание ответа...")

        if parallel:
            # Параллельное выполнение запросов
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(query_model, model['id'], prompt, api_key): model
                    for model in selected_models
                }

                for future in futures:
                    model = futures[future]
                    result = future.result()
                    results[model['id']] = result

                    # Обновляем контейнер с результатом
                    if result['success']:
                        containers[model['id']].markdown(result['content'])
                        st.toast(f"✅ {model['name']} ответил за {result['time']:.1f}с")
                    else:
                        containers[model['id']].error(f"❌ {result['error']}")
        else:
            # Последовательное выполнение
            progress = st.progress(0)
            for idx, model in enumerate(selected_models):
                result = query_model(model['id'], prompt, api_key)
                results[model['id']] = result

                if result['success']:
                    containers[model['id']].markdown(result['content'])
                else:
                    containers[model['id']].error(f"❌ {result['error']}")

                progress.progress((idx + 1) / len(selected_models))
            progress.empty()

        # Статистика
        st.markdown("---")
        st.subheader("📈 Статистика")

        successful = sum(1 for r in results.values() if r['success'])
        failed = len(results) - successful
        avg_time = sum(r['time'] for r in results.values() if r['success']) / max(successful, 1)

        stat_cols = st.columns(4)
        stat_cols[0].metric("✅ Успешно", successful)
        stat_cols[1].metric("❌ Ошибки", failed)
        stat_cols[2].metric("⏱️ Среднее время", f"{avg_time:.1f}с")
        stat_cols[3].metric("📊 Всего моделей", len(results))


if __name__ == "__main__":
    main()
