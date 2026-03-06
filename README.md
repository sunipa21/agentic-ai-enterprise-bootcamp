# Day 1 – Context Handling Fundamentals in LLM Systems

## Objective

Understand why naive LLM invocation fails in real-world systems and how structured message-based invocation preserves context.

This exercise demonstrates:

- Why string-based prompts are stateless
- How structured message history maintains conversation memory
- Why context management is critical in production-grade AI systems

---

## Problem Statement

When interacting with an LLM using simple string prompts, each call is isolated.

Example:

```
User: "We are building an AI system for processing medical insurance claims."
User: "What are the main risks in this system?"
```

In naive implementation, the model cannot properly answer the second question because the context about medical insurance claims is not passed to the second invocation.

Enterprise AI systems must manage structured conversation history to enable:

- Multi-turn conversations
- Session-based interactions
- Agent memory
- Context-aware decision making

---

## 🏗 Implementation Overview

This project contains two demonstrations:

### 1. Naive Stateless Invocation
- Each LLM call is independent
- No memory retention
- Context is lost between calls
- Fails to properly address follow-up questions without context

### 2. Message-Based Stateful Invocation
- Structured message history is maintained
- Prior conversation is passed to model
- Multi-turn memory works reliably
- Provides proper context for all follow-up queries

---

## 🚀 How to Run

To execute the application, run:

```bash
python app.py
```

This will demonstrate both naive stateless invocation and the message-based context fix.

---

## 📋 Requirements

- Python 3.8+
- OpenAI API key (set in `.env` file)
- Dependencies listed in `requirements.txt`

## 🔧 Installation

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` file with your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   python app.py
   ```

---

## 📊 Output

The application demonstrates:
1. **Context Break**: Naive invocation where second question lacks context
2. **Context Fix**: Message-based invocation preserving full conversation history
3. **Enterprise Implications**: Why message history is critical for production AI systems