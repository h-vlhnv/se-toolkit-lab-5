## This lab

### This lab - TODO

- install nodejs via nix
- update caddy port - should be the biggest
- goal: interact with the database, not just observe
- rename: the autochecker -> `Autochecker`
- describe full vm setup
- use mdsh for nixpkgs hashes
- connect by remote ssh - check your ip to understand where you are
- [?] Connect by ssh - can't find ssh config in Linux
- should a section in a sequence of steps assume the previous step?
- lab-prompts.md? - prompts for agents
  - bundle all instructions for task 1 in a readable doc
- don't use claude-specific words in skills
- Use consistent API token (or key?) naming
- Rename app -> backend
- Rename `APP_` -> `BACK_`
- Add `FRONT_` suffix for front-end variables
- Always provide links to variables from .env.docker.secret
- `.env.local.example`
- Student: ask agent why these tests
- move generic troubleshooting sections to wiki
- autochecker: check file submissions size
- Is this true? "# This solution won't work outside the University network."
- fix adjacent links
- GitHub Pages with good full-text search
- `Remote-SSH: Connect to Host...`
- what is X -> About X?
- setup-simple.md - a simpler version of setup.md
- Move constants with `CONST_` prefix from `.env.docker.example` to `.env.const`
- conventions: prompt engineering steps:
  - don't provide a ready prompt first.
  - hint at what to think about when writing a prompt.
  - provide the prompt under a spoiler.
- Move ideas to the instructors/ideas.md.
- Use instructors/meetings just for storing meeting notes, not for the lab design.
- skill /issue
- russian version
- Prompt students to use `skill commit`
- setup: The instructions aren't guaranteed to work outside of Linux or macOS. This is why we require to use WSL
- coding-agents.md 1000 requests - each file read is a request
- Fix config after the migration from the older repo
- Migrate relevant parts of inno-se/the-guide
- grafana later when we have multiple apps
- qwen and ssh
- conventions: new sentence on a new line
- line break between instructions
- line break after curl command
- fix /fix-file-by-conventions skill: write title instead of cross-out in the task report.
- skill: terms not explained in the wiki
- conventions: indented note is a quotation

### This lab - DONE

- [x] CRLF - fixed via .gitattributes
- [x] switch to typescript
- [x] kebab case most of the time - via claude conventions
- [x] direnv allow
- [x] caddy container has frontend?
- [x] in git-workflow.md, explain how to git pull
- [x] update db name: lab-4 to db-lab-4
- [x] update server name: lab-4 to postgres-lab-4
- [x] reuse claude skills for qwen

     Symlinked QWEN.md to CLAUDE.md and .qwen to .claude.
     However, the qwen extension can't use skills.
     Only the CLI version can.

- [x] api.md
- [x] use postgres 18
- [x] conventions include self-checks
- [?] add and remove cors
- [x] Rename `API_TOKEN` -> `API_KEY`
- [x] install extensions not in WSL but earlier and then reopen in WSL
        Mentioned in the docs on instaling the recommended extensions
- [x] move from testing.md to quality-assurance.md
- [x] Conventions:
        Tasks must create controlled environment.
        Even AI steps.
- [x] Conventions - outcomes must be verifiable using the autochecker
- [x] Move constants to `.env.docker.example` with `CONST_` prefix
- [x] convention - visualize task
- [x] use a more neutral IP address in examples
        Use `---1`
- [x] review via conventions - check conceptual problems from the educational and practical point of view.
- [x] conventions:

  files must stay in sync:

  - dotenv-docker-secret.md with .env.docker.example
  - dotenv-tests-unit-secret.md with .env.tests.unit.example
  - dotenv-tests-e2e-secret.md with .env.tests.e2e.example
  - pyproject-toml.md with pyproject.toml
- [x] report "Which conceptual problems are in ..." then which conventions are violated
        Have the skill /review-task-conceptual
- [x] Run through autochecker note at the end of each task
        Have a convention now

## Next labs

- test front
- Implement a Status page <https://status.claude.com/>
   Must be a separate service that checks health. Grafana?
- caddy https
- include logging
- script for database backup
- deploy via github actions
- use pnpm or bun
