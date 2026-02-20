#!/usr/bin/env python3
"""Telegram personal AI manager.

Reads personal knowledge from baza.txt and replies in Telegram DMs from the account owner.
"""

from __future__ import annotations

import json
import logging
import asyncio
from pathlib import Path
from typing import Any

import requests
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError

BASE_DIR = Path(__file__).resolve().parent
APP_CONFIG_PATH = BASE_DIR / "app_config.json"
PROVIDER_CONFIG_PATH = BASE_DIR / "provider_config.json"
KNOWLEDGE_PATH = BASE_DIR / "baza.txt"
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"
DEFAULT_MODEL = "openai/gpt-4o-mini"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ai_manager")


class ConfigError(RuntimeError):
    """Raised when config files are missing or invalid."""



def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Не найден файл: {path.name}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Ошибка JSON в файле {path.name}: {exc}") from exc



def load_knowledge(path: Path) -> str:
    if not path.exists():
        raise ConfigError(
            f"Не найден файл {path.name}. Создайте его и заполните базой знаний."
        )

    data = path.read_text(encoding="utf-8").strip()
    if not data:
        raise ConfigError(f"Файл {path.name} пуст. Добавьте туда информацию о себе.")
    return data


def load_system_prompt(path: Path) -> str:
    if not path.exists():
        raise ConfigError(
            f"Не найден файл {path.name}. Создайте его и добавьте системный промпт."
        )

    data = path.read_text(encoding="utf-8").strip()
    if not data:
        raise ConfigError(
            f"Файл {path.name} пуст. Добавьте туда инструкции для стиля общения."
        )
    return data



def build_system_prompt(base_prompt: str, knowledge: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "Дополнительные правила: пиши ответы в личке Telegram от первого лица, как "
        "владелец аккаунта. Никогда не называй себя ассистентом или ботом. "
        "Если информации в базе недостаточно, мягко уточняй детали.\n\n"
        "БАЗА ЗНАНИЙ ВЛАДЕЛЬЦА АККАУНТА:\n"
        f"{knowledge}"
    )



def call_llm(
    provider: dict[str, Any],
    system_prompt: str,
    user_message: str,
    history: list[dict[str, str]],
) -> str:
    base_url = provider.get("base_url", "").rstrip("/")
    api_key = provider.get("api_key", "")
    model = provider.get("model", DEFAULT_MODEL)
    endpoint = provider.get("endpoint", "/chat/completions")
    timeout_sec = int(provider.get("timeout_sec", 60))

    if not base_url:
        raise ConfigError("В provider_config.json поле 'base_url' обязательно.")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    messages = [{"role": "system", "content": system_prompt}, *history]
    messages.append({"role": "user", "content": user_message})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": provider.get("temperature", 0.7),
    }

    # Совместимость с некоторыми провайдерами, требующими max_tokens.
    if "max_tokens" in provider:
        payload["max_tokens"] = provider["max_tokens"]

    response = requests.post(
        f"{base_url}{endpoint}",
        headers=headers,
        json=payload,
        timeout=timeout_sec,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Ошибка провайдера LLM: {response.status_code} {response.text[:500]}"
        )

    body = response.json()

    # OpenAI/OpenRouter форматы:
    try:
        return body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Не удалось прочитать ответ модели. Проверьте формат API и endpoint."
        ) from exc



def trim_history(history: list[dict[str, str]], max_messages: int) -> list[dict[str, str]]:
    if max_messages <= 0:
        return []
    return history[-max_messages:]



async def main() -> None:
    app_cfg = load_json(APP_CONFIG_PATH)
    provider_cfg = load_json(PROVIDER_CONFIG_PATH)
    knowledge = load_knowledge(KNOWLEDGE_PATH)
    base_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)
    startup_system_prompt = build_system_prompt(base_prompt, knowledge)

    api_id = app_cfg.get("api_id")
    api_hash = app_cfg.get("api_hash")
    phone = app_cfg.get("phone")
    session_name = app_cfg.get("session_name", "my_telegram_session")

    if not api_id or not api_hash or not phone:
        raise ConfigError(
            "В app_config.json обязательны поля: api_id, api_hash, phone."
        )

    max_history_messages = int(app_cfg.get("max_history_messages", 20))
    memory: dict[int, list[dict[str, str]]] = {}

    client = TelegramClient(session_name, int(api_id), str(api_hash))

    logger.info("Запуск клиента Telegram...")
    await client.connect()

    if not await client.is_user_authorized():
        logger.info("Нужна авторизация. Отправляем код на номер %s", phone)
        await client.send_code_request(phone)
        code = input("Введите код из Telegram: ").strip()
        try:
            await client.sign_in(phone=phone, code=code)
        except SessionPasswordNeededError:
            password = input("Введите облачный пароль (2FA): ").strip()
            await client.sign_in(password=password)

    @client.on(events.NewMessage(incoming=True))
    async def on_message(event: events.NewMessage.Event) -> None:
        if not event.is_private:
            return

        sender = await event.get_sender()
        if sender and getattr(sender, "is_self", False):
            return

        text = (event.raw_text or "").strip()
        if not text:
            return

        user_id = event.sender_id
        chat_history = memory.setdefault(user_id, [])
        chat_history = trim_history(chat_history, max_history_messages)

        try:
            current_knowledge = load_knowledge(KNOWLEDGE_PATH)
            current_base_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)
            current_system_prompt = build_system_prompt(
                current_base_prompt,
                current_knowledge,
            )
        except ConfigError:
            logger.exception(
                "Ошибка чтения baza.txt/system_prompt.txt. Использую промпт из старта."
            )
            current_system_prompt = startup_system_prompt

        try:
            answer = call_llm(provider_cfg, current_system_prompt, text, chat_history)
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка при обращении к LLM")
            await event.reply(
                "Извини, сейчас не могу ответить (ошибка модели). Попробуй позже."
            )
            return

        chat_history.append({"role": "user", "content": text})
        chat_history.append({"role": "assistant", "content": answer})
        memory[user_id] = trim_history(chat_history, max_history_messages)

        await event.reply(answer)

    logger.info("Скрипт активен. Ожидаю входящие ЛС... Ctrl+C для остановки.")
    await client.run_until_disconnected()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ConfigError as exc:
        logger.error("Ошибка конфигурации: %s", exc)
    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C")
