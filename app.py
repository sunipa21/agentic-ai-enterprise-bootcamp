"""
Day 2 Assignment – Routing with LangGraph (Tier-Based Support Flow)

This module demonstrates:

1. Typed state management with LangGraph (SupportState)
2. Explicit routing logic based on user tier (vip vs standard)
3. Conditional edge handling with add_conditional_edges
4. Simple but production-minded workflow design

The routing is auditable, testable, and ready for enterprise use.
"""

from typing import TypedDict, Annotated
from operator import add
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

# =====================================================
# Environment Setup
# =====================================================

load_dotenv()

# =====================================================
# STATE DEFINITION
# =====================================================


class SupportState(TypedDict):
    """
    Typed state for the support routing workflow.

    Fields:
        messages: Conversation history (accumulates with operator.add)
        should_escalate: Boolean flag indicating escalation need
        issue_type: Categorization of the support issue
        user_tier: Customer tier ("vip" or "standard")
    """

    messages: Annotated[list[BaseMessage], add]
    should_escalate: bool
    issue_type: str
    user_tier: str


# =====================================================
# ROUTING LOGIC
# =====================================================


def route_by_tier(state: SupportState) -> str:
    """
    Route the support request based on user tier.

    Routing decisions:
        - "vip" users → "vip_path" (fast-track, no escalation)
        - All others → "standard_path" (normal flow, may escalate)

    Args:
        state: Current support state

    Returns:
        Route name as string ("vip_path" or "standard_path")
    """
    if state.get("user_tier") == "vip":
        return "vip_path"
    return "standard_path"


# =====================================================
# NODES (Workflow Steps)
# =====================================================


def check_user_tier_node(state: SupportState) -> dict:
    """
    Determine if the user is a VIP or standard customer.

    Implementation:
        - Checks the first message for keywords like "vip" or "premium"
        - Simple mock for demonstration (in production: database lookup)

    Args:
        state: Current support state

    Returns:
        Dictionary with updated user_tier
    """
    first_message = state["messages"][0].content.lower()

    if "vip" in first_message or "premium" in first_message:
        user_tier = "vip"
    else:
        user_tier = "standard"

    print(f"✓ check_user_tier_node: Classified as '{user_tier}'")

    return {"user_tier": user_tier}


def vip_agent_node(state: SupportState) -> dict:
    """
    Handle VIP customer path.

    Characteristics:
        - Fast-track handling
        - No escalation needed
        - Premium service tier

    Args:
        state: Current support state

    Returns:
        Dictionary with VIP-specific handling flags
    """
    print("→ vip_agent_node: Handling VIP customer with priority service")

    return {
        "should_escalate": False,
    }


def standard_agent_node(state: SupportState) -> dict:
    """
    Handle standard customer path.

    Characteristics:
        - Normal workflow
        - May escalate if needed
        - Standard service tier

    Args:
        state: Current support state

    Returns:
        Dictionary with standard handling flags
    """
    print("→ standard_agent_node: Handling standard customer")

    return {
        "should_escalate": True,
    }


# =====================================================
# GRAPH CONSTRUCTION
# =====================================================


def build_graph():
    """
    Construct and compile the support routing workflow graph.

    Graph structure:
        1. check_tier (entry point)
        2. [conditional split]
           - vip_agent (VIP path)
           - standard_agent (standard path)
        3. END

    Returns:
        Compiled LangGraph StateGraph
    """
    workflow = StateGraph(SupportState)

    # Add nodes
    workflow.add_node("check_tier", check_user_tier_node)
    workflow.add_node("vip_agent", vip_agent_node)
    workflow.add_node("standard_agent", standard_agent_node)

    # Set entry point
    workflow.set_entry_point("check_tier")

    # Add conditional routing from check_tier based on route_by_tier function
    workflow.add_conditional_edges(
        "check_tier",
        route_by_tier,
        {
            "vip_path": "vip_agent",
            "standard_path": "standard_agent",
        },
    )

    # Add edges to END
    workflow.add_edge("vip_agent", END)
    workflow.add_edge("standard_agent", END)

    return workflow.compile()


# =====================================================
# EXECUTION
# =====================================================


def main() -> None:
    """
    Main entry point demonstrating the routing workflow.

    Executes two test cases:
        1. VIP customer flow
        2. Standard customer flow

    Outputs:
        - user_tier and should_escalate for each flow
        - Printed routing decisions for visibility
    """
    print("\n" + "=" * 70)
    print("DAY 2 ASSIGNMENT – ROUTING WITH LANGGRAPH")
    print("=" * 70 + "\n")

    # Build the graph
    graph = build_graph()

    # --------- TEST CASE 1: VIP CUSTOMER ---------
    print("[TEST 1] VIP Customer Flow")
    print("-" * 70)

    vip_result = graph.invoke(
        {
            "messages": [
                HumanMessage(content="I'm a VIP customer, please check my order status")
            ],
            "should_escalate": False,
            "issue_type": "",
            "user_tier": "",
        }
    )

    print(
        f"\nResult → user_tier: '{vip_result.get('user_tier')}' | should_escalate: {vip_result.get('should_escalate')}"
    )
    print()

    # --------- TEST CASE 2: STANDARD CUSTOMER ---------
    print("[TEST 2] Standard Customer Flow")
    print("-" * 70)

    standard_result = graph.invoke(
        {
            "messages": [HumanMessage(content="I need help with my order")],
            "should_escalate": False,
            "issue_type": "",
            "user_tier": "",
        }
    )

    print(
        f"\nResult → user_tier: '{standard_result.get('user_tier')}' | should_escalate: {standard_result.get('should_escalate')}"
    )
    print()

    # --------- SUMMARY ---------
    print("=" * 70)
    print("ROUTING LOGIC SUMMARY")
    print("=" * 70)
    print("✓ VIP path: user_tier='vip', should_escalate=False (fast-track)")
    print("✓ Standard path: user_tier='standard', should_escalate=True (may escalate)")
    print()


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
