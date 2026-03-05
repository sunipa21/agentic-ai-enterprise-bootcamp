"""
Day 1 – Context Handling Fundamentals

This module demonstrates:

1. Naive LLM invocation (stateless)
2. Message-based invocation (stateful)
3. Structured logging with latency and token tracking
4. Session-aware invocation pattern

This mirrors real-world enterprise AI service logging patterns.
"""

import logging
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

# =====================================================
# Environment Setup
# =====================================================

load_dotenv()

# =====================================================
# Logging Configuration (Structured JSON Logging)
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("llm_agent")

# =====================================================
# LLM Initialization
# =====================================================

llm_openai = ChatOpenAI(model="gpt-4.1-nano")

# =====================================================
# GLOBAL Conversation (Stateful Example)
# =====================================================

conversation = [
    SystemMessage(content="You are a concise, professional, and friendly assistant.")
]

# =====================================================
# Utility
# =====================================================


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60 + "\n")


# =====================================================
# Stateful Logged Invoke (Message-Based)
# =====================================================


def logged_invoke(user_message: str, session_id: str):
    """
    Stateful LLM invocation using structured message history.
    """

    start_time = datetime.now()

    conversation.append(HumanMessage(content=user_message))

    logger.info(
        json.dumps(
            {
                "event": "llm_call_start",
                "session_id": session_id,
                "timestamp": start_time.isoformat(),
                "user_message": user_message,
                "model": "gpt-4.1-nano",
                "mode": "stateful",
            }
        )
    )

    try:
        response = llm_openai.invoke(conversation)
        latency = (datetime.now() - start_time).total_seconds()

        conversation.append(AIMessage(content=response.content))

        logger.info(
            json.dumps(
                {
                    "event": "llm_call_success",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "latency_seconds": latency,
                    "response": response.content,
                    "output_tokens": response.usage_metadata.get("output_tokens"),
                    "input_tokens": response.usage_metadata.get("input_tokens"),
                }
            )
        )

        return response.content

    except Exception as e:
        latency = (datetime.now() - start_time).total_seconds()

        logger.error(
            json.dumps(
                {
                    "event": "llm_call_error",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "latency_seconds": latency,
                    "error": str(e),
                }
            )
        )

        raise e


# =====================================================
# Stateless Invoke (Naive)
# =====================================================


def naive_invoke(user_message: str, session_id: str):
    """
    Stateless LLM invocation.
    Does NOT retain previous conversation context.
    """

    start_time = datetime.now()

    logger.info(
        json.dumps(
            {
                "event": "llm_call_start",
                "session_id": session_id,
                "timestamp": start_time.isoformat(),
                "user_message": user_message,
                "model": "gpt-4.1-nano",
                "mode": "stateless",
            }
        )
    )

    try:
        response = llm_openai.invoke(
            [
                SystemMessage(content="You are a concise, professional assistant."),
                HumanMessage(content=user_message),
            ]
        )

        latency = (datetime.now() - start_time).total_seconds()

        logger.info(
            json.dumps(
                {
                    "event": "llm_call_success",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "latency_seconds": latency,
                    "response": response.content,
                    "output_tokens": response.usage_metadata.get("output_tokens"),
                    "input_tokens": response.usage_metadata.get("input_tokens"),
                }
            )
        )

        return response.content

    except Exception as e:
        latency = (datetime.now() - start_time).total_seconds()

        logger.error(
            json.dumps(
                {
                    "event": "llm_call_error",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "latency_seconds": latency,
                    "error": str(e),
                }
            )
        )

        raise e


# =====================================================
# Demo Execution
# =====================================================


def run_demo():
    print_separator("NAIVE STATELESS DEMO")

    session_stateless = str(uuid.uuid4())

    print("User: My name is Sunil.")
    print("Assistant:", naive_invoke("My name is Sunil.", session_stateless))

    print("\nUser: What is my name?")
    print("Assistant:", naive_invoke("What is my name?", session_stateless))

    print_separator("STATEFUL MESSAGE-BASED DEMO")

    session_stateful = str(uuid.uuid4())

    print("User: My name is Sunil.")
    print("Assistant:", logged_invoke("My name is Sunil.", session_stateful))

    print("\nUser: What is my name?")
    print("Assistant:", logged_invoke("What is my name?", session_stateful))

    print_separator("ENTERPRISE OBSERVATION")

    print("""
Stateless Invocation:
- Each call is independent.
- No conversation memory.
- Fails for multi-turn dialogue.

Stateful Invocation:
- Structured message history preserved.
- Enables contextual continuity.
- Required for AI Agents, RAG systems,
  enterprise chat systems, and session-based architectures.
""")


# =====================================================
# Main Entry
# =====================================================

if __name__ == "__main__":
    logger.info("Application started")
    run_demo()
    logger.info("Application finished")

"""
=====================================================
Reflection:
=====================================================
1. Why did string-based invocation fail?

   In string-based (stateless) invocation, each LLM call is isolated.
   The model receives only the current prompt and does not have access
   to previous conversation history. Therefore, it cannot recall
   prior user inputs such as name, preferences, or prior context.

2. Why does message-based invocation work?

   Message-based invocation explicitly passes structured conversation
   history (SystemMessage, HumanMessage, AIMessage) back to the model
   on each call. This preserves conversational state and enables
   multi-turn contextual continuity.

3. What would break in a production AI system if we ignore message history?

   - Multi-turn conversations would fail.
   - AI agents would lose memory of previous tool outputs.
   - RAG systems would not maintain conversational grounding.
   - Session-based user interactions would become inconsistent.
   - Enterprise chat systems would produce incorrect or incoherent responses.

   In short, ignoring message history makes stateful AI systems impossible.
"""
