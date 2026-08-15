"""Sprint 2 CLI entrypoint.

Now shows:
- Argument quality (strong/weak/fallacy)
- Score breakdown (logic/evidence/clarity)
- Handler decision
- AI counterargument adapted to quality

Sprint 1 showed: just the AI counterargument
Sprint 2 shows:  full graph output with classification + scoring
"""

from src.observability.langsmith_setup import setup_langsmith
from src.observability.logfire_setup import setup_logfire

langsmith_ok = setup_langsmith()
logfire_ok   = setup_logfire()

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
    print(f"   Observability Active")
    print(f"  LangSmith: {'enabled' if langsmith_ok else 'disabled'}")
    print(f"  Logfire:   {'enabled' if logfire_ok else 'disabled'}")
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
        # Sprint 7 — show guardrail feedback
        if not result.get("input_guard_passed"):
            print(f"\n ARGUMENT BLOCKED")
            print("─" * 40)
            print(f"  {result.get('input_guard_reason', 'Input failed validation.')}")
            print("─" * 40)
            debate_history.append(("human", user_argument))
            debate_history.append(("assistant", result.get("ai_response", "")))
            turn += 1
            continue

        # Show any output guard warnings
        output_results = result.get("output_guard_results", [])
        warnings = [r for r in output_results if r.get("action") == "warn"]
        if warnings:
            print(f"\n  ⚠️  Note: {warnings[0].get('reason', '')}")


            
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



        summary = result.get("debate_summary", "")
        similar = result.get("similar_past_args", [])
        if summary:
            print(f"\nDEBATE SUMMARY SO FAR")
            print("─" * 40)
            # truncate for display
            display_summary = summary[:200] + "..." if len(summary) > 200 else summary
            print(f"  {display_summary}")
        if similar:
            print(f"\nSIMILAR PAST ARGUMENT DETECTED")
            print("─" * 40)
            best = similar[0]
            display_arg = best['argument'][:80]
            print(f"  Turn {best['turn_number']}: \"{display_arg}...\"")
            print(f"  Quality: {best['quality'].upper()} | Score: {best['score']}/10 | Similarity: {best['similarity']}")

        # ── AI counterargument ────────────────────────────────────────────────
        print(f"\nAI COUNTERARGUMENT ({ai_side.upper()})")
        print(f"{'─'*40}")
        ai_response = result.get("ai_response", "")
        print(ai_response)

        # Add AI response to history
        debate_history.append(("assistant", ai_response))
        turn += 1

        # Sprint 9 — turn evaluation
        eval_score    = result.get("turn_eval_score", 0)
        eval_grade    = result.get("turn_eval_grade", "")
        eval_feedback = result.get("turn_eval_feedback", "")

        if eval_grade:
            grade_icons = {
                "excellent": "★★★★",
                "good":      "★★★☆",
                "average":   "★★☆☆",
                "poor":      "★☆☆☆",
            }
            print(f"\nAI RESPONSE QUALITY")
            print("─" * 40)
            print(
                f"  Grade:    {grade_icons.get(eval_grade, '')} "
                f"{eval_grade.upper()} ({eval_score}/10)"
            )
            if eval_feedback:
                print(f"  Feedback: {eval_feedback}")

    # Sprint 9 — session evaluation on debate end
    print(f"\n{'='*65}")
    print(f"  Debate Complete — {turn - 1} turns")
    print(f"{'='*65}")

    session_eval = service.evaluate_session()

    if session_eval:
        print(f"\nSESSION EVALUATION")
        print("─" * 40)

        direction_icons = {
            "improving":          "📈 IMPROVING",
            "declining":          "📉 DECLINING",
            "stable":             "➡️  STABLE",
            "insufficient_data":  "📊 INSUFFICIENT DATA",
        }
        print(
            f"  Progress:     "
            f"{direction_icons.get(session_eval.user_improvement, '')}"
        )
        print(
            f"  First half:   "
            f"{session_eval.avg_score_first_half:.1f}/10 avg"
        )
        print(
            f"  Second half:  "
            f"{session_eval.avg_score_second_half:.1f}/10 avg"
        )
        print(f"  Score trend:  {session_eval.score_trend}")
        print(f"  Overall grade: {session_eval.overall_grade.upper()}")
        print()
        print(f"  Strong args:  {session_eval.strong_count}")
        print(f"  Weak args:    {session_eval.weak_count}")
        print(f"  Fallacies:    {session_eval.fallacy_count}")
        print(f"  Best turn:    Turn {session_eval.best_turn}")
        print(f"  Worst turn:   Turn {session_eval.worst_turn}")
        print()
        print(f"  Tokens used:  {session_eval.total_tokens:,}")
        print(f"  Cost:         ${session_eval.total_cost_usd:.6f}")
        print(f"  Cache hits:   {session_eval.cache_hits}")
        print()
        print(f"COACHING ADVICE")
        print("─" * 40)
        print(f"  {session_eval.improvement_advice}")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    run_debate()