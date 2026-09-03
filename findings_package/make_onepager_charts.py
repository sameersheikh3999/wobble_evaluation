"""Two house-style charts for the one-page wobble summary."""
import sys, os
sys.path.insert(0, r"C:\Users\HP\.claude\skills\notion-board\reference")
import economist_chart as ec

OUT = r"c:\Users\HP\Music\AI Projects\wobble_evaluation\findings_package\charts"
os.makedirs(OUT, exist_ok=True)

TEAL, ORANGE, RED = ec.TEAL, ec.ORANGE, ec.ACCENT_RED

# ---- Chart 1: the wobble is concentrated in seven indicators -----------------
labels = [
    "C4  Equitable Participation",
    "D5  On-Task Behaviour",
    "B2  Lesson Structure & Sequence",
    "D4  Student Confidence",
    "C6  Classroom Management",
    "B9  Time on Task",
    "D6  Use of Learning Materials",
    "The other 29 indicators (average)",
]
values = [18.2, 15.8, 13.3, 11.5, 11.5, 11.0, 10.2, 3.4]
palette = [RED] * 7 + [TEAL]

fig = ec.bar_h(
    title="Seven indicators produce half the disagreement",
    subtitle=("How often a re-scored verdict contradicts the majority. "
              "Seven of 36 indicators account for 48% of all disagreement; "
              "the remaining 29 average 3.4%."),
    labels=labels, values=values, palette=palette,
    xaxis_label="disagreement rate between identical re-scorings (%)",
    value_fmt="{:.1f}%",
    source="10 classroom observations, scored 10 times each on two scales - 800 scoring passes",
)
fig.set_size_inches(13, 5.4)   # wider and shorter: vertical space is the constraint
ec.save_chart(fig, os.path.join(OUT, "onepager_1_concentration.png"))

# ---- Chart 2: removing them, without changing the scorer --------------------
# One metric only. The earlier version put a 5% rate and an 87-point reliability
# score on the same axis, which flattened the bars that carried the point.
fig2 = ec.bar_h(
    title="Removing the seven cuts disagreement by a third",
    labels=["All 37 indicators", "The 30 that hold"],
    values=[5.4, 3.5],
    palette=[ORANGE, TEAL],
    xaxis_label="average disagreement rate (%)",
    value_fmt="{:.1f}%",
    source="Same 10 observations, re-analysed - no additional scoring",
)
fig2.set_size_inches(13, 3.4)  # one-line subtitle, so a short figure is safe
ec.save_chart(fig2, os.path.join(OUT, "onepager_2_effect.png"))
print("charts written to", OUT)
