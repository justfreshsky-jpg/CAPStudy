# CAPStudy

CAP cadet Achievement Test practice quiz generator. Live at <https://capstudy.freshskyai.com>.

Pick achievement (Curry through Spaatz) + subject (Leadership Lab or Aerospace Education) → get 10 fresh practice questions with explanations. Multiple choice, instant grading, and an 80% practice target. Stateless — no scores stored across sessions.

Flask app using the pinned `freshsky-common` package for the education privacy-restricted LLM chain, security headers, and abuse limits. The request contains only the selected achievement, subject, and question count; no names, CAP member IDs, rosters, contact details, or other personal data belong in the app.

Quiz API responses are private/no-store and noindexed. Likely identifiers are rejected before a provider call, model output is validated against a strict quiz schema, and choices use native radio controls with keyboard and screen-reader feedback. Scores remain browser-local and are not stored in an application database.
