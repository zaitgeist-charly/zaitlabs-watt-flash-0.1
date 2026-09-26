# watt-flash-0.1

Small, ultra-fast Jev-style decision model (`noul`, `choice`, `score`) -> option probabilities.

## Use

```bash
git clone https://github.com/zaitgeist-charly/zaitlabs-watt-flash-0.1 && cd zaitlabs-watt-flash-0.1
uv sync
uv run python -m watt_flash '{"state": "Doppelt belastet, bitte erstatten.",
  "questions": {"refund": {"type": "noul", "instructions": "Refund requested?"}}}'
# {"refund": [P(false), P(true)]}
```

Weights are fetched from [huggingface.co/zaitlabs/watt-flash-0.1](https://huggingface.co/zaitlabs/watt-flash-0.1) on first run.

```python
from watt_flash import WattFlash
wf = WattFlash()
wf.decide(state, questions)                       # {qid: probabilities}
wf.decide_batch([(state1, qs1), (state2, qs2)])   # several requests, one forward pass
```

Free test API:

```bash
curl https://api.inference.zaitlabs.com/ai/run \
  --header "Content-Type: application/json" \
  --data '{"model": "watt-flash-0.1", "input": {"state": "Help! My payouts have been failing for 3 days.",
           "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}}'
```

## Model

- mmBERT-small backbone, 140M parameters, 8,192-token context.
- Training: multilingual question corpus, 490,000 examples, 20 languages.

## Strengths and weaknesses

- 3-13x faster than [Laya](https://github.com/NandhaKishorM/laya) at 1k-4k state tokens, 5 questions per request.
- Multi-lingual trained.
- Can't compete with models >= 1B

## Benchmarks

Pending.

## License

Apache 2.0
