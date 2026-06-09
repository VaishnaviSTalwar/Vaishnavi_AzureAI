"""
Mood-Aware Chatbot
==================
- Tracks a mood score that shifts based on positive / negative words in user input
- Adjusts reply style: Cheerful  (score > 0.3)
                       Neutral   (-0.3 ≤ score ≤ 0.3)
                       Concerned (score < -0.3)
- Displays a live mood bar and per-turn mood history
- Pure Python — no ML libraries required
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import re
import time
import random

# ─────────────────────────────────────────────
# 1. SENTIMENT LEXICON
# ─────────────────────────────────────────────

POSITIVE_WORDS = {
    "happy", "great", "good", "awesome", "fantastic", "love", "excellent",
    "wonderful", "excited", "joy", "joyful", "amazing", "brilliant", "fun",
    "glad", "pleased", "cheerful", "thrilled", "superb", "nice", "beautiful",
    "grateful", "thankful", "positive", "cool", "delightful", "perfect",
    "smile", "laugh", "confident", "hopeful", "enjoy", "success", "win",
    "winning", "best", "well", "fine", "okay", "yes", "yay", "wow",
}

NEGATIVE_WORDS = {
    "sad", "bad", "terrible", "awful", "horrible", "hate", "angry", "upset",
    "depressed", "miserable", "frustrated", "annoyed", "worried", "anxious",
    "stressed", "tired", "exhausted", "bored", "lonely", "disappointed",
    "unhappy", "failure", "fail", "lost", "worst", "worse", "pain", "hurt",
    "crying", "cry", "scared", "fear", "helpless", "hopeless", "broken",
    "sorry", "regret", "mistake", "wrong", "no", "not", "never", "none",
}

# Amplifiers & negators
AMPLIFIERS = {"very", "really", "so", "extremely", "absolutely", "totally", "super"}
NEGATORS   = {"not", "never", "no", "don't", "doesn't", "didn't", "won't",
               "can't", "cannot", "hardly", "barely"}


# ─────────────────────────────────────────────
# 2. SENTIMENT ANALYSER
# ─────────────────────────────────────────────

def analyse_sentiment(text: str) -> float:
    """
    Returns a sentiment delta in [-1, 1] for the given text.
    Considers amplifiers (+50 % weight) and negators (flip sign).
    """
    tokens = re.findall(r"[a-z']+", text.lower())
    score  = 0.0
    i = 0
    while i < len(tokens):
        word = tokens[i]
        # Look back for negator in a 2-word window
        negated   = any(tokens[max(0, i-2):i+1][j] in NEGATORS
                        for j in range(min(2, i)))
        amplified = i > 0 and tokens[i-1] in AMPLIFIERS
        weight    = 1.5 if amplified else 1.0

        if word in POSITIVE_WORDS:
            score += weight * (-1 if negated else 1)
        elif word in NEGATIVE_WORDS:
            score += weight * (1 if negated else -1)   # negated negative → positive
        i += 1

    # Normalise by word count so short sentences aren't penalised
    word_count = max(len(tokens), 1)
    return max(-1.0, min(1.0, score / (word_count ** 0.5)))


# ─────────────────────────────────────────────
# 3. MOOD TRACKER
# ─────────────────────────────────────────────

class MoodTracker:
    """
    Maintains a running mood score in [-1, 1] with exponential smoothing.
    """
    SMOOTHING = 0.35          # weight given to newest observation

    def __init__(self):
        self.score   = 0.0    # starts neutral
        self.history = []     # list of (turn, score) tuples

    def update(self, delta: float, turn: int):
        self.score = (1 - self.SMOOTHING) * self.score + self.SMOOTHING * delta
        self.score = max(-1.0, min(1.0, self.score))
        self.history.append((turn, round(self.score, 3)))

    @property
    def label(self) -> str:
        if self.score > 0.3:
            return "Cheerful"
        elif self.score < -0.3:
            return "Concerned"
        return "Neutral"

    @property
    def emoji(self) -> str:
        return {"Cheerful": "😊", "Neutral": "😐", "Concerned": "😟"}[self.label]

    def mood_bar(self, width: int = 30) -> str:
        """ASCII bar: |----[■]-----|  centred at 0."""
        pos      = int((self.score + 1) / 2 * width)   # map [-1,1] → [0, width]
        pos      = max(0, min(width - 1, pos))
        mid      = width // 2
        bar      = ["-"] * width
        bar[mid] = "|"                                  # zero mark
        bar[pos] = "■"
        colour   = "\033[92m" if self.label == "Cheerful" else \
                   "\033[91m" if self.label == "Concerned" else "\033[93m"
        reset    = "\033[0m"
        return f"{colour}[{''.join(bar)}]{reset}"


# ─────────────────────────────────────────────
# 4. REPLY TEMPLATES
# ─────────────────────────────────────────────

CHEERFUL_REPLIES = [
    "That's wonderful to hear! 😄 {echo} — keep that energy going!",
    "Love your enthusiasm! ✨ Sounds like things are going great.",
    "Awesome! 🎉 You're on a roll — what else is on your mind?",
    "That totally makes sense and I'm here for it! 🌟",
    "You're radiating good vibes — keep it up! 😊",
]

NEUTRAL_REPLIES = [
    "Got it. {echo} — tell me more if you'd like.",
    "Understood. I'm here and listening.",
    "Thanks for sharing. What else is on your mind?",
    "I see. Feel free to elaborate whenever you're ready.",
    "Noted! Anything specific you'd like to talk through?",
]

CONCERNED_REPLIES = [
    "I hear you, and it sounds like things have been tough. 💙 Want to talk about it?",
    "That doesn't sound easy at all. I'm here — no rush.",
    "I'm sorry you're going through that. 😟 What's been weighing on you most?",
    "It's okay to feel this way. Take your time — I'm listening.",
    "That sounds really hard. Is there anything I can do to help you think it through?",
]

TEMPLATES = {
    "Cheerful":  CHEERFUL_REPLIES,
    "Neutral":   NEUTRAL_REPLIES,
    "Concerned": CONCERNED_REPLIES,
}


def generate_reply(mood: MoodTracker, user_text: str) -> str:
    template = random.choice(TEMPLATES[mood.label])
    # Pull a short echo from the user's text (first 6 words)
    words = user_text.strip().split()
    echo  = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
    return template.replace("{echo}", f'"{echo}"')


# ─────────────────────────────────────────────
# 5. MOOD HISTORY CHART (ASCII)
# ─────────────────────────────────────────────

def print_mood_history(history: list):
    if not history:
        return
    print("\n" + "=" * 55)
    print("  MOOD SCORE HISTORY  (turn-by-turn)")
    print("=" * 55)

    turns  = [h[0] for h in history]
    scores = [h[1] for h in history]

    chart_h = 10   # rows
    chart_w = max(len(turns) * 4, 20)

    # Normalise scores to [0, chart_h-1]
    def to_row(s):
        return int((1 - (s + 1) / 2) * (chart_h - 1))

    grid = [[" "] * chart_w for _ in range(chart_h)]

    # Draw axes
    mid_row = chart_h // 2
    for col in range(chart_w):
        grid[mid_row][col] = "·"
    for row in range(chart_h):
        grid[row][0] = "│"
    grid[mid_row][0] = "┼"

    # Plot points and connect with lines
    positions = []
    for i, (t, s) in enumerate(history):
        col = 2 + i * 4
        row = to_row(s)
        if 0 <= col < chart_w and 0 <= row < chart_h:
            grid[row][col] = "●"
            positions.append((row, col))

    # Simple vertical connectors between consecutive points
    for i in range(len(positions) - 1):
        r1, c1 = positions[i]
        r2, c2 = positions[i + 1]
        if c1 < chart_w - 1:
            mid_c = (c1 + c2) // 2
            mid_r = (r1 + r2) // 2
            if 0 <= mid_c < chart_w and 0 <= mid_r < chart_h and grid[mid_r][mid_c] == " ":
                grid[mid_r][mid_c] = "╌"

    print(f"  +1.0 ┐")
    for row_i, row in enumerate(grid):
        label = ""
        if row_i == 0:
            label = "+1.0"
        elif row_i == mid_row:
            label = " 0.0"
        elif row_i == chart_h - 1:
            label = "-1.0"
        print(f"  {label:>4s} │ {''.join(row)}")
    print(f"  -1.0 ┘")

    # X-axis labels
    x_labels = "       "
    for i, (t, _) in enumerate(history):
        x_labels += f"T{t:<3d}"
    print(x_labels)

    print("\n  Turn-by-turn scores:")
    for t, s in history:
        mood_label = (
            "Cheerful 😊" if s > 0.3 else
            "Concerned 😟" if s < -0.3 else
            "Neutral  😐"
        )
        bar = "▓" * int(abs(s) * 10)
        print(f"    Turn {t:>2d}: {s:+.3f}  [{bar:<10s}]  {mood_label}")

    print("=" * 55)


# ─────────────────────────────────────────────
# 6. CHATBOT MAIN LOOP
# ─────────────────────────────────────────────

INTRO = """
╔═══════════════════════════════════════════════════╗
║          MOOD-AWARE CHATBOT  🤖                   ║
║                                                   ║
║  Chat freely — I'll track your mood in real-time. ║
║  Type  'history'  to see the mood chart.          ║
║  Type  'quit'     to exit.                        ║
╚═══════════════════════════════════════════════════╝
"""

def run_chatbot():
    print(INTRO)
    mood  = MoodTracker()
    turn  = 0
    max_turns = 3

    while True:
        if turn >= max_turns:
            print("\n[Session automatically concluded after 3 turns]")
            break
            
        # ── Prompt ─────────────────────────────────
        try:
            prompt_str = f"You [Press Enter for default: '{DEMO_SCRIPT[turn]}']: "
            user_input = input(prompt_str).strip()
            if not user_input:
                user_input = DEMO_SCRIPT[turn]
                print(f"Using default: {user_input}")
        except (EOFError, KeyboardInterrupt):
            print("\n\n[Session ended by user]")
            break

        if user_input.lower() == "quit":
            print("\nBot: Thanks for chatting! Take care. 👋")
            break

        if user_input.lower() == "history":
            print_mood_history(mood.history)
            continue

        # ── Sentiment → mood update ─────────────────
        turn += 1
        delta = analyse_sentiment(user_input)
        mood.update(delta, turn)

        # ── Reply ───────────────────────────────────
        reply = generate_reply(mood, user_input)
        time.sleep(0.3)   # tiny pause for realism

        # ── Display ─────────────────────────────────
        print(f"\n{'─'*52}")
        print(f"  Mood: {mood.emoji} {mood.label:<10s}  Score: {mood.score:+.3f}")
        print(f"  {mood.mood_bar()}")
        print(f"{'─'*52}")
        print(f"Bot: {reply}\n")

    # ── End-of-session summary ──────────────────────
    if mood.history:
        print_mood_history(mood.history)
        avg  = sum(s for _, s in mood.history) / len(mood.history)
        peak = max(mood.history, key=lambda x: x[1])
        low  = min(mood.history, key=lambda x: x[1])
        print(f"\n  Session Summary")
        print(f"  ───────────────")
        print(f"  Turns       : {turn}")
        print(f"  Avg score   : {avg:+.3f}")
        print(f"  Peak mood   : {peak[1]:+.3f}  (turn {peak[0]})")
        print(f"  Lowest mood : {low[1]:+.3f}  (turn {low[0]})")
        print()


# ─────────────────────────────────────────────
# 7. DEMO MODE  (non-interactive)
# ─────────────────────────────────────────────

DEMO_SCRIPT = [
    "Hey! I'm feeling really great today, everything is awesome!",
    "Had an amazing morning, went for a run and enjoyed the sunshine.",
    "But the afternoon was horrible — got some bad news at work.",
    "I'm so stressed and tired, I don't know what to do.",
    "Talking about it is helping a bit though. You're really nice.",
    "I think I'll be okay. Things can get better, right?",
    "Thanks! I feel much better now. Happy to have chatted!",
]

def run_demo():
    print(INTRO)
    print("[ DEMO MODE — simulating a conversation ]\n")
    mood = MoodTracker()

    for turn, text in enumerate(DEMO_SCRIPT, start=1):
        print(f"You: {text}")
        delta = analyse_sentiment(text)
        mood.update(delta, turn)

        reply = generate_reply(mood, text)
        time.sleep(0.5)

        print(f"\n{'─'*52}")
        print(f"  Mood: {mood.emoji} {mood.label:<10s}  Score: {mood.score:+.3f}")
        print(f"  {mood.mood_bar()}")
        print(f"{'─'*52}")
        print(f"Bot: {reply}\n")

    print_mood_history(mood.history)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        # Try interactive; fall back to demo if stdin is not a tty
        if sys.stdin.isatty():
            run_chatbot()
        else:
            print("[Non-interactive environment detected — running demo mode]\n")
            run_demo()
