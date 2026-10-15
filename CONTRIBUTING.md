# Contributing (team guide)

This is a team assignment: everyone makes a meaningful contribution and reviews a
teammate's work. The flow below is the same whether you write code by hand or with
an agent such as Hermes or Claude Code. If you use an agent, point it at
`AGENTS.md` first.

## One-time setup

1. Accept the collaborator invitation (GitHub → the repository → Invitations, or the
   email). Until then you can read the repo but not push.
2. Clone and install (details in `README.md`):
   ```bash
   git clone https://github.com/jones-benjam/MBAX_build_a_course_assistant.git
   cd MBAX_build_a_course_assistant
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env     # then fill in the class endpoint values; never commit .env
   ./run_tests.sh           # should pass offline without any key
   ```
   In `.env`, also change `MOCK_SERVICES=1` to `MOCK_SERVICES=0`; otherwise the app
   runs on offline stand-ins. On Windows, use `.venv\Scripts\activate`, and run the
   tests with `set MOCK_SERVICES=1` then `python -m pytest -q`.
3. Install LibreOffice if you will work with PowerPoint files.

## Picking up work

- Work is organised as GitHub **issues**; each has a "Done when" checklist. Assign
  yourself to an issue (or ask in the team chat) before starting so two people do
  not do the same thing.
- Create a **branch** from `main` for the issue:
  ```bash
  git checkout main && git pull
  git checkout -b issue-14-screenshots
  ```
- Commit as you go with messages that say what and why. Push the branch:
  ```bash
  git push -u origin issue-14-screenshots
  ```
- Open a **pull request** (PR) on GitHub from your branch into `main`. In the
  description, copy the issue's checklist and tick what is done, and write
  `Closes #14` so the issue closes when the PR merges.
- Ask a teammate to **review**: they read the diff, run `./run_tests.sh`, and
  either approve or request changes. The author never merges their own PR without
  a review.
- After merge, delete the branch and move to the next issue.

### Reviewer checklist

- [ ] The change does what the issue's "Done when" list says.
- [ ] `./run_tests.sh` and `pyflakes` pass on your machine.
- [ ] No key, real endpoint URL, `.env` or course file in the diff (`git grep API_KEY=`).
- [ ] Docs touched by the change (README, HANDOFF, ENDPOINTS) are updated.
- [ ] Leave the review on GitHub (Approve or Request changes), not just in chat.

## Terms, briefly

- **Branch:** a separate line of work so `main` always stays runnable.
- **Commit:** a saved snapshot with a message; small and frequent is good.
- **Pull request:** a proposal to merge a branch into `main`, with discussion and review attached.
- **Review:** a teammate's check of the change before it lands.
- **Issue:** a task or bug with a checklist; links the discussion to the PR that resolves it.
- **Mock services:** `MOCK_SERVICES=1` runs the app and tests with built-in stand-ins for the class models, so no key or network is needed.

## Rules that protect the team

- Never commit `.env`, the API key, the class endpoint note, or the Canvas course
  files. `.gitignore` already excludes `.env` and `course_materials/`.
- Week 4 slides 12 and 13 show the class endpoint key. Do not screenshot them or
  paste answers that quote them anywhere in the repo or on GitHub.
- Keep tests and `pyflakes` green before pushing.
- If you find a bug, open an issue with steps to reproduce and what you expected,
  even if you plan to fix it yourself.
- Prefer small PRs; a reviewer can read 200 lines carefully, not 2000.
