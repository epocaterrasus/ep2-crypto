# Session Protocol: How Development Works Across Claude Sessions

## The Problem

- Each Claude session has limited context (~200K tokens usable)
- A single sprint has 5-15 files to create, each with tests
- Context compression loses detail — you can't rely on "remembering" what was discussed
- Mid-sprint handoffs are the #1 source of bugs and inconsistencies

## Solution: Sprint Tickets + Session Log

### Structure

Each sprint is broken into **tickets** (atomic units of work). Each ticket:
- Can be completed in a single session (or less)
- Has a clear input (what exists) and output (what to create)
- Has a verification command (how to prove it's done)
- Is tracked in `PROGRESS.md` with status

### Session Lifecycle

```
START OF SESSION:
1. Read CLAUDE.md (auto-loaded)
2. Read PROGRESS.md → find current ticket
3. Read the ticket's "Context" section → understand what exists
4. Do the work
5. Run verification commands
6. Update PROGRESS.md with what was done
7. If ticket complete, mark [x] and note any issues for next ticket

END OF SESSION (or context getting large):
1. Update PROGRESS.md with:
   - What was completed
   - What was NOT completed
   - Any decisions made during this session
   - Any issues discovered
   - The EXACT next step (not vague — specific file + function)
2. Commit the code with meaningful message
3. If mid-ticket, write a HANDOFF note in PROGRESS.md
```

### Context Management Rules

1. **Never read research files into context unless actively implementing from them** — they're 10-50KB each and eat context fast
2. **Read only the sections you need** — use `offset` and `limit` on Read tool
3. **One ticket per session is fine** — quality over speed
4. **If context is getting large (>150K tokens), wrap up** — update PROGRESS.md and stop
5. **Each session should start fresh** — don't rely on conversation history, rely on files

### Sprint-to-Ticket Breakdown

Each sprint in SPRINTS.md gets decomposed into tickets in PROGRESS.md before work begins.

Example for Sprint 1:
```
## Sprint 1: Foundation

### S1-T1: Project structure and package init
- Create src/ep2_crypto/__init__.py and all subpackage __init__.py files
- Verify: uv run python -c "import ep2_crypto"

### S1-T2: Configuration module
- Create src/ep2_crypto/config.py with Pydantic Settings
- Verify: uv run pytest tests/test_config.py

### S1-T3: Logging setup
- Create src/ep2_crypto/logging.py with structlog JSON config
- Verify: uv run python -c "from ep2_crypto.logging import setup; setup()"

### S1-T4: Database schema
- Create src/ep2_crypto/db/schema.py
- Verify: uv run pytest tests/test_db/test_schema.py

### S1-T5: Database repository
- Create src/ep2_crypto/db/repository.py
- Verify: grep -r "f'" src/ep2_crypto/db/ returns nothing
```

### How to Start a New Sprint

1. Decompose the sprint into tickets in PROGRESS.md
2. Each ticket gets: description, files to create, research reference, verification
3. Estimate: 1-3 tickets per session depending on complexity
4. Start with ticket 1

### How to Handle "I'm Stuck"

If a ticket is harder than expected:
1. Write what you learned in PROGRESS.md
2. Mark ticket as [~] (in progress) with a note
3. Move on to the next ticket if possible (don't block)
4. Come back to it in a fresh session with more context budget

## Data Sources Update

### Polymarket Integration (added post-research)

Polymarket provides prediction market data for crypto-related events:
- **API**: `https://gamma-api.polymarket.com/markets`
- **Relevant markets**: "Will BTC be above $X by Y date?", regulatory outcomes
- **Use as**: background regime feature (daily resolution), NOT real-time signal
- **Integration point**: Sprint 5 (cross-market features), polled every 30 min
- **Key insight from research**: prediction market prices can differ from options-implied probabilities — useful as an alternative sentiment/probability source

### All Data Sources (Complete List)

| Source | Frequency | Sprint | Priority |
|--------|-----------|--------|----------|
| Binance WS: klines, depth, aggTrades | Real-time | S2 | Critical |
| Bybit: OI, funding, liquidations | 5min/real-time | S2 | Critical |
| ccxt: ETH/USDT alongside BTC | Real-time | S2 | High |
| yfinance: NQ, Gold, DXY | 5-min poll (15-min delayed) | S5 | High |
| mempool.space: whale txs | Real-time WS | S5 | Medium |
| Binance: long/short ratio | 5-min poll | S2 | Medium |
| Deribit: options IV, put/call | 5-min poll | S5 | Medium |
| Alternative.me: Fear & Greed | 30-min poll | S5 | Low |
| Polymarket: BTC-related markets | 30-min poll | S5 | Low |
| Coinbase: BTC-USD (premium calc) | Real-time | S5 | Medium |
