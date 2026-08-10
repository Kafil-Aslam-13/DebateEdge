"""Sprint 2 CLI entrypoint.

Now shows:
- Argument quality (strong/weak/fallacy)
- Score breakdown (logic/evidence/clarity)
- Handler decision
- AI counterargument adapted to quality

Sprint 1 showed: just the AI counterargument
Sprint 2 shows:  full graph output with classification + scoring
"""

from src.core.config import get_settings
from src.core.logger import get_logger
from src.services.debate_service import DebateService

logger = get_logger(__name__)

QUALITY_COLORS = {
    "strong":  "STRONG",
    "weak":    "WEAK",
    "fallacy": "FALLACY",
}


def print_score_bar(score: int, max_score: int = 10) -> str:
    """Visual score bar for terminal output."""
    filled = int((score / max_score) * 20)
    bar = "█" * filled + "░" * (20 - filled)
    return f"[{bar}] {score}/{max_score}"


def run_debate():
    settings = get_settings()
    service = DebateService()

    print(f"\n{'='*65}")
    print(f"  {settings.app_name} — AI Debate & Argument Coach")
    print(f"  Sprint 2: LangGraph Workflow Active")
    print(f"{'='*65}\n")

    # Topic
    topic = input("Enter a debate topic: ").strip()
    if not topic:
        print("Topic cannot be empty.")
        return

    # Side
    print("\nWhich side do you want to argue?")
    print("  1. For")
    print("  2. Against")
    choice = input("Enter 1 or 2: ").strip()
    user_side = "for" if choice == "1" else "against"
    ai_side = "against" if user_side == "for" else "for"

    print(f"\nYou argue: {user_side.upper()} | AI argues: {ai_side.upper()}")
    print(f"Topic: {topic}\n")
    print("-" * 65)

    # AI opens
    print("\nAI Opening Statement:")
    print("-" * 65)
    opening = service.open_debate(topic, user_side)
    print(opening)
    print("-" * 65)

    # Debate loop
    debate_history = []
    turn = 1

    while turn <= settings.max_turns:
        print(f"\n{'─'*65}")
        print(f"  Turn {turn}/{settings.max_turns}")
        print(f"{'─'*65}")

        user_argument = input("\nYour argument: ").strip()

        if not user_argument:
            print("Please enter an argument.")
            continue

        if user_argument.lower() in ["quit", "exit", "q"]:
            print("\nDebate ended.")
            break

        # Add user message to history
        debate_history.append(("human", user_argument))

        print("\nAnalysing your argument...\n")

        # Process through LangGraph
        result = service.process_argument(
            topic=topic,
            user_side=user_side,
            user_argument=user_argument,
            debate_history=debate_history[:-1],
            turn_number=turn,
        )

        # ── Show analysis results ─────────────────────────────────────────────
        quality = result.get("argument_quality", "unknown")
        score = result.get("argument_score", 0)
        breakdown = result.get("score_breakdown", {})
        reasoning = result.get("quality_reasoning", "")

        print(f"ARGUMENT ANALYSIS")
        print(f"{'─'*40}")
        print(f"  Quality:   {QUALITY_COLORS.get(quality, quality.upper())}")
        print(f"  Score:     {print_score_bar(score)}")

        if breakdown:
            print(f"  Logic:     {breakdown.get('logic', '?')}/10")
            print(f"  Evidence:  {breakdown.get('evidence', '?')}/10")
            print(f"  Clarity:   {breakdown.get('clarity', '?')}/10")

        if reasoning:
            print(f"  Feedback:  {reasoning}")

        # Sprint 3 — show fallacy details
        if result.get("contains_fallacy"):
            print(f"\nFALLACY DETECTED")
            print(f"{'─'*40}")
            print(f"  Name:       {result.get('fallacy_name', '').upper()}")
            print(f"  Severity:   {result.get('fallacy_severity', '').upper()}")
            if result.get("fallacy_explanation"):
                print(f"  Explained:  {result.get('fallacy_explanation')}")
            if result.get("fallacy_correction", "none") != "none":
                print(f"  Fix it:     {result.get('fallacy_correction')}")

        # ── AI counterargument ────────────────────────────────────────────────
        print(f"\nAI COUNTERARGUMENT ({ai_side.upper()})")
        print(f"{'─'*40}")
        ai_response = result.get("ai_response", "")
        print(ai_response)

        # Add AI response to history
        debate_history.append(("assistant", ai_response))
        turn += 1

    print(f"\n{'='*65}")
    print(f"  Debate Complete — {turn - 1} turns")
    print(f"  Fallacy detection (Sprint 3) and memory (Sprint 4) coming next.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_debate()