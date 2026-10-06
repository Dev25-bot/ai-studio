# Benchmark 20261007-023549

### sentiment

| task | model | n | accuracy | macro_f1 | confusion | latency_ms/sample | device |
|---|---|---|---|---|---|---|---|
| sentiment | distilbert-base-uncased-finetuned-sst-2-english | 3 | 1.0 | 1.0 | ['NEGATIVE', 'POSITIVE'] [[1, 0], [0, 2]] | 578.8 | mps |

### next_word

| task | model | n | perplexity | eval_s | first_token_ms | tokens/s | device |
|---|---|---|---|---|---|---|---|
| next_word | gpt2 | 5 | 30.22 | 0.48 | 67.7 | 34.9 | mps |
