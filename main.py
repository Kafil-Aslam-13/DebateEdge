"""Sprint 1 CLI entrypoint — basic debate working.

Usage:
    python main.py

This is the simplest possible test of the debate system.
Replace with FastAPI (Sprint 12) and Streamlit (Sprint 13) later.
"""

from src.core.config import get_settings
from src.core.logger import get_logger
from src.services.debate_service import DebateService

logger = get_logger(__name__)


def run_debate():
    settings = get_settings()
    service = DebateService()

    print(f"\n{'='*60}")
    print(f"  Welcome to {settings.app_name}")
    print(f"  AI Debate and Argument Coach")
    print(f"{'='*60}\n")

    # Get topic
    topic = input("Enter a debate topic: ").strip()
    if not topic:
        print("Topic cannot be empty.")
        return

    # Get user side
    print("\nWhich side do you want to argue?")
    print("  1. For")
    print("  2. Against")
    choice = input("Enter 1 or 2: ").strip()
    user_side = "for" if choice == "1" else "against"
    ai_side = "against" if user_side == "for" else "for"

    print(f"\nYou are arguing: {user_side.upper()}")
    print(f"AI is arguing:   {ai_side.upper()}")
    print(f"\nTopic: {topic}\n")
    print("-" * 60)

    # AI opens the debate
    print("\nAI Opening Statement:")
    print("-" * 60)
    opening = service.open_debate(topic, user_side)
    print(opening)
    print("-" * 60)

    # Debate loop
    debate_history = []
    turn = 1

    while turn <= settings.max_turns:
        print(f"\n[Turn {turn}/{settings.max_turns}]")
        user_argument = input("\nYour argument: ").strip()

        if not user_argument:
            print("Please enter an argument.")
            continue

        if user_argument.lower() in ["quit", "exit", "q"]:
            print("\nDebate ended.")
            break

        # Add user message to history
        debate_history.append(("human", user_argument))

        # Get AI counterargument
        print("\nAI Counterargument:")
        print("-" * 60)
        ai_response = service.argue(
            topic=topic,
            user_side=user_side,
            user_argument=user_argument,
            debate_history=debate_history[:-1],  # history before current message
        )
        print(ai_response)
        print("-" * 60)

        # Add AI response to history
        debate_history.append(("assistant", ai_response))
        turn += 1

    print(f"\n{'='*60}")
    print("Debate complete. Coaching and scoring coming in Sprint 3.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_debate()