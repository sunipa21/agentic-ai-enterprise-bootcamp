# Day 2 – Routing with LangGraph (Tier-Based Support Flow)

## Objective

Design a **minimal but production-minded LangGraph workflow** that routes users to different support paths based on **user tier** (VIP vs. standard).

This assignment demonstrates:

- Typed state management with LangGraph (`SupportState`)
- Explicit, auditable routing logic (`route_by_tier`)
- Conditional edge handling with `add_conditional_edges`
- A clean, testable workflow foundation

---

## Problem Statement

In real-world systems, routing decisions must be **explicit and auditable**. Without proper routing logic, support workflows become chaotic and inconsistent.

Example scenario:

```
User 1: "I'm a VIP customer, I need urgent help"
User 2: "I need help with my order"
```

Without tier-based routing:
- Both requests treated equally
- VIP customers don't receive priority handling
- No distinction between urgent and standard issues
- Business logic is scattered across the code

Enterprise support systems must implement **structured routing** to enable:

- Tier-based prioritization
- Consistent handling across different paths
- Auditable decision logic
- Scalable workflow architecture

---

## 🏗 Implementation Overview

This project implements a support routing system with two distinct paths:

### 1. Tier Detection
- Analyzes incoming support request
- Identifies customer tier (VIP or standard)
- Routes to appropriate handling path

### 2. VIP Path
- Fast-track handling
- No escalation needed
- Premium service tier

### 3. Standard Path
- Normal workflow
- May escalate if needed
- Standard service tier

---

## 🤖 Why OpenAI API is Not Used

This implementation uses **mock nodes** instead of LLM API calls for the following reasons:

1. **Assignment Focus**: The core requirement is demonstrating **routing logic**, not LLM capabilities
2. **Cost Efficiency**: Avoids unnecessary API calls during testing and development
3. **Simplicity**: Focuses on the foundation—typed state, explicit routing, and graph structure
4. **Production Pattern**: In real deployments, this routing layer would orchestrate different LLM calls based on tier

The assignment explicitly states:
> "You can call an LLM here if you want. For the assignment it is fine to just set a friendly VIP response. The grader only checks for **correct routing behavior**, not your prompt wording."

**Future Enhancement**: LLM calls can be easily added to `vip_agent_node()` and `standard_agent_node()` once the routing foundation is solid.

---

## 🚀 How to Run

To execute the application, run:

```bash
python app.py
```

This will execute two test cases (VIP and standard customer) and display the routing results.

---

## 📋 Requirements

- Python 3.8+
- OpenAI API key (set in `.env` file)
- Dependencies listed in `requirements.txt`

---

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

## ⚠️ Important

**Do NOT commit `.env` to version control.** The `.env` file contains your API key and must be excluded in `.gitignore`.

---

## 📊 Output

The application demonstrates:

- **Tier Detection**: Identifies customer tier from message content
- **Routing Logic**: Routes VIP and standard customers to appropriate paths
- **State Management**: Tracks `user_tier` and `should_escalate` flags
- **Auditable Results**: Clear visibility into routing decisions for both test cases