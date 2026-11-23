# Brief: tiibrarian on the two Tiinys, through woollama (2026-11-23)

## STAR summary

**Situation.** tiibrarian (this repo) is the cited-answer layer over the offline ZIM shelf. It retrieves passages, asks a model for claims, and keeps only the claims whose quoted span is word for word on the cited page. A session in the Maplewick repo (a study game for Teague's daughter Carys) wants it to make short, grounded reading cards, for example on Roman aqueducts, lipids, and the French passé composé.

**Task.** Make tiibrarian run reliably on the two Tiiny devices through woollama, the model gateway on dobby. Fix the two defects found below, on a branch from `origin/main`.

**Action (so far, by the Maplewick session, Claude Opus 5.5).** It ran `zimshelf.py --mode vector --device` under Slurm with both device licenses. It found that the default answer budget is too small for the reasoning model on puffskein, and that `--mode fts` crashes.

**Result (true now).** Vector retrieval through pygmy works. With `TIIBRARIAN_MAX_TOKENS` raised, 3 of 3 test questions return grounded claims. **Caveat:** these runs used branch `agy/retire-two-stage-int8`, which is 9 commits behind `origin/main` and has no commits of its own (it is merged). Main changed retrieval (`095b5a0` ranks through `corpus.retrieve`; `a4519b4` routes Wikipedia to its own DB). Re-measure on main before you trust any figure below.

## Status

Two bugs and one configuration finding. No design is needed. Do not run brainstorming. Start a branch from `origin/main`. This brief file is not committed: it sits untracked in the checkout. Commit it on your branch.

## Your prompt

You are the tiibrarian maintainer for this track. Do these, in order:

1. **Make a branch from `origin/main`.** The checkout is on a stale, merged branch. Other worktrees under `~/Projects/tiibrarian-trees` hold `agy/*` branches; do not disturb them.
2. **Reproduce both defects on main**, under Slurm (see the device rules below), before you change code.
3. **Fix the answer budget.** `synth.py:43` defaults `MAX_TOKENS` to 700. `tiiny2/default` is a reasoning model: it spends the budget on thinking, returns `finish_reason: length` with empty `content`, and tiibrarian then abstains with `kept: []` and `dropped: []`. That looks the same as "the library does not cover this", which is wrong and silent. Two parts:
   - Raise the default, or make it depend on the model. Your call; say why.
   - **Fail loud:** when `finish_reason` is `length` and the content holds no parseable claims, say "the model ran out of tokens". Do not report it as an abstention. Write the test first, and see it fail.
4. **Fix the fts crash.** `zimshelf.py --mode fts --archive vikidia_en_all_maxi_2026-06 "how did Roman aqueducts carry water"` raised `KeyError: 'chunk'` at `zimshelf.py:134` (`page_ident`), after retrieval returned hits. It is possibly from `3ff1463` (the passage ordinal), but this is not checked. Main still reads `hit["chunk"]` at line 134. fts is the only route to Gutenberg, which has no vectors.
5. **Document the woollama route** in the README: the env vars below, and the token file (by path, never the value).
6. Run the hermetic tests (blq if this repo is registered; otherwise `srun -p ci`). Open a PR. Report to Teague.

## Measured (2026-11-22/23, longbottom, Slurm with `-L tiiny-pygmy,tiiny-puffskein -w longbottom`)

Environment that worked (from the woollama session pi-a6, which measured the embedder at 1024 dims, L2 norm 91.9, matching the index):

```bash
export TIINY_AUTH_KEY="$(sed -n 's/^WOOLLAMA_DOBBY_TOKEN=//p' ~/.config/woollama/dobby.env)"
export TIINY_EMBED_URL=https://woollama.aidr.me/v1/embeddings
export TIINY_EMBED_MODEL=tiiny-embed/Qwen/Qwen3-Embedding-0.6B   # pygmy; the ONLY one that matches the index
export TIINY_CHAT_URL=https://woollama.aidr.me/v1/chat/completions
export TIINY_CHAT_MODEL=tiiny2/default                           # puffskein, Qwen3.6-35B-A3B-Turbo
export TIIBRARIAN_ZIM_DIRS=/srv/logical/zimshelf/Library/zim      # the default /srv/logical/portable/zim does not exist on longbottom
export TIIBRARIAN_MAX_TOKENS=6000
```

Budget probe (one short page, same prompt):

| max_tokens | finish | content chars | reasoning chars | completion tokens |
|---|---|---|---|---|
| 700 | length | 0 | 2813 | 700 |
| 3000 | stop | 227 (one correct claim) | 5195 | 1399 |

End-to-end, `--mode vector --topk 5 --device`:

| Question | Budget | Result | Time |
|---|---|---|---|
| How did Roman aqueducts carry water to cities? | 700 | abstain, 0 kept, 0 dropped | 30 s |
| same | 3000 | 3 kept (Wikibooks Roman_Culture/aqueducts) | 53 s |
| What do lipids do in the body? | 3000 | abstain, 0/0 | 60 s |
| same | 6000 | 4 kept | 49 s |
| How do you form the passé composé in French? | 3000 | abstain, 0/0 | 70 s |
| same | 6000 | 5 kept (Wikibooks French grammar) | 84 s |

Retrieval found on-topic pages for all three questions at every budget. Vector index `~/corpus/zim-corpora.duckdb` (20.7 GB, table `pages`, 1.8M passages): Wikibooks 334k, Simple Wikipedia 326k, Wikipedia medicine 215k, Khan Academy 34k, Vikidia 8k, and others. **Not** indexed: Gutenberg, the Stardew wiki, full Wikipedia.

## Limits agreed up front

- **Device rules (pi-a6):** take the Slurm licenses `tiiny-pygmy` (embedding) and `tiiny-puffskein` (synthesis). Load the `slurm` skill first. Embed one query at a time: pygmy wedges under concurrent load. A 503 from puffskein means "busy" (Teague's pi sessions share it), so retry later. Keep requests short. If pygmy embeddings start to time out, stop and tell Teague. Never power-cycle it or log in to it.
- Do not use `tiiny2-embed/Qwen/Qwen3-Embedding-4b`: it is a different encoder.
- Never print or commit the token. Do not use the device keys in `~/.local/tiiny/*.json`.
- Keep the grounding contract (`CONTRACT.md`) as it is. The fix makes failures visible. It does not loosen what counts as grounded.

## First question to settle

None that forks the design. If the budget fix wants a per-model default, decide it in the PR and say why.

## Links

- Originating session: Maplewick, `db07c8ad` (claude.ai/code/session_01NSRdqbXs1ZjmmjLuEwDut5). Reply to the session named `maplewick-84` when the PR is up.
- Scripts and raw output from these runs: `/tmp/claude-1000/-srv-logical-workspace-home-teague-Projects-maplewick/db07c8ad-45d0-4195-8cb2-2e6beb48ad29/scratchpad/` (`ask.sh`, `ask6k.sh`, `probe.sh`, `cards.out`, `cards6k.out`). It is a scratchpad, so copy what you need.
- Files: `synth.py:43,140-156`, `zimshelf.py:125-140,320-330`, `corpus.py:75-110`, `CONTRACT.md`.
