# 3-minute screen recording: what to say and show (no slides)

0:00 to 0:20  Terminal/GitHub. "Client wants a fraud flag with accuracy above 97%. Here is what I tried, changed and threw away."
0:20 to 0:55  Notebook or EVIDENCE.md. "Only 1 in 30 recent claims is fraud, so flagging nothing is already about 97%. I threw the accuracy goal away and used fraud stopped per claim checked." Show the 96.9% vs 95.8% lines.
0:55 to 1:25  Data traps. "681 duplicate claim numbers removed. 202 undecided cases dropped, not counted as genuine. Partner history only looks back 45 days so it cannot cheat."
1:25 to 2:00  The surprise. "I first trained on all history. It scored worse than random on May and June (0.22), because the 1 May no-inspection rule changed what fraud looks like. So I tested on June only and weighted recent claims 10 times." Show the old-regime AUC line.
2:00 to 2:30  Live demo: start the service, open the screen, click the two examples, read the reasons aloud.
2:30 to 3:00  Honest limits. "Only 36 frauds since May, so numbers are rough. Net gain is small, about Rs 2,300 a month; the bigger lever is putting a few partners back on inspection."
Things I tried and discarded (say one): text from descriptions and inspector notes (no signal), logistic regression (worse than boosting on June), training on pre-May only (worse than random).
