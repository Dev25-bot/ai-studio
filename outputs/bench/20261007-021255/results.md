# Benchmark 20261007-021255

### summarization

| task | model | n | rouge1 | rouge2 | rougeL | len_ratio | latency_s (median) | device |
|---|---|---|---|---|---|---|---|---|
| summarization | sshleifer/distilbart-cnn-12-6 | 20 | 36.03 | 16.83 | 28.27 | 0.1331 | 2.023 | mps |

### next_word

| task | model | n | perplexity | eval_s | first_token_ms | tokens/s | device |
|---|---|---|---|---|---|---|---|
| next_word | gpt2 | 20 | 30.6 | 2.61 | 668.5 | 87.3 | mps |
