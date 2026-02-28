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

User: "My name is Sunil."
User: "What is my name?"

In naive implementation, the model cannot answer correctly because previous context is not passed.

Enterprise AI systems must manage structured conversation history to enable:

- Multi-turn conversations
- Session-based interactions
- Agent memory
- Context-aware decision making

---

## 🏗 Implementation Overview

This project contains two demonstrations:

### 1 Naive Stateless Invocation
- Each LLM call is independent
- No memory retention
- Context is lost between calls

### 2 Message-Based Stateful Invocation
- Structured message history is maintained
- Prior conversation is passed to model
- Multi-turn memory works reliably