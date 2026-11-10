# 90-second demo script

| Time | Do | Say |
|---|---|---|
| 0:00–0:12 | Show the app home, Inventory tab | "Bars track inventory on clipboards, so they run out of things mid-shift. BarBack lets staff just *say* what happened, and an LLM agent updates the books." |
| 0:12–0:30 | Chat: **"We just got 2 kegs of Bud Light and 3 bottles of Jack"** → open the Agent trace | "The model returns structured JSON. It's validated, then executed. Here are its reasoning and both database actions, and the stock went up." |
| 0:30–0:45 | Chat: **"Poured 14 shots of Jameson and 6 of Patron tonight"** | "It handles bar math: 14 shots is about 0.62 of a liter bottle. It also warns when something drops below par." |
| 0:45–1:00 | Chat: **"Draft orders for everything below par and post a heads-up to the team"** → Files tab, open the draft | "That's a multi-step plan: a database report, then a file write. The team notification is an external API call, so it…" |
| 1:00–1:12 | Approve panel appears → click **Approve** | "…pauses for human approval. Same for deletes and resets." |
| 1:12–1:22 | Chat: **"Delete Fireball"** → **Decline** | "Decline, and nothing happens. The model acknowledges it." |
| 1:22–1:30 | Audit log tab | "Every input, LLM call, validation and action is logged. It works with Claude, GPT or Gemini, with retries and self-correction if the model's output is invalid." |

Tip: before recording, run each prompt once so you know what the model says.
