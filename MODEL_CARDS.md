# Model cards & licensing notes

All checkpoints are downloaded from the Hugging Face Hub on first use and cached in `~/.cache/huggingface`.
Swap any of them in `configs/app.yaml`. Always check the upstream model card before production use.

| Task | Default checkpoint | Params | Licence | Notes / known limitations |
|---|---|---|---|---|
| Image generation | [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo) | ~0.9 B | Stability AI Non-Commercial Research Community licence | Adversarial-diffusion distilled SD 2.1; 1–4 steps, 512². Not for commercial use without a Stability licence. Can reproduce social biases; text rendering is poor. |
| Image generation (alt) | [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) | ~0.9 B | CreativeML OpenRAIL-M | 20–50 steps, guidance ≈ 7.5. Use-based restrictions in the licence. |
| Summarization | [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6) | 306 M | Apache-2.0 | Distilled BART-large-CNN; news-style extractive-leaning summaries; 1024-token window (longer inputs are chunked). |
| Summarization (alt) | `facebook/bart-large-cnn`, `google/flan-t5-base` | 406 M / 248 M | MIT / Apache-2.0 | FLAN-T5 is instruction-tuned and more abstractive but scores lower ROUGE on CNN/DM. |
| Sentiment | [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | 67 M | Apache-2.0 | Binary POSITIVE/NEGATIVE only — neutral text is forced into one class. Known to show nationality/identity biases. |
| Next-word | [`gpt2`](https://huggingface.co/openai-community/gpt2) | 124 M | MIT | Trained on WebText (2019); may produce biased, offensive or false text. |
| Translation | [`Helsinki-NLP/opus-mt-en-de`](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) | 74 M | CC-BY-4.0 | One direction per checkpoint; swap for other `opus-mt-<src>-<tgt>` pairs. |
| NER | [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER) | 108 M | MIT | CoNLL-2003 tags: PER, ORG, LOC, MISC. English news domain. |
| Q&A | [`deepset/roberta-base-squad2`](https://huggingface.co/deepset/roberta-base-squad2) | 124 M | CC-BY-4.0 | Extractive: the answer must be a span of the context. |
| Image classification | [`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224) | 86 M | Apache-2.0 | ImageNet-1k labels only. |
| CLIPScore (eval) | [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) | 151 M | MIT | Used only for evaluation. |

## Datasets
| Task | Dataset | Licence |
|---|---|---|
| Summarization | CNN/DailyMail 3.0.0 test subset (`data/summarization/`) | Apache-2.0 |
| Sentiment | SST-2 validation (GLUE) (`data/sentiment/`) | see GLUE / SST terms (research use) |
| Next-word | WikiText-2 raw test (`data/next_word/`) | CC BY-SA 3.0 |
| Image prompts | Hand-written `assets/prompts.csv` | Project licence |
