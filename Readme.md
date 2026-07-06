# Semantic Image Search Engine (CLIP + Streamlit)

Search a set of images using natural language queries — no labels, no training.
Built on OpenAI's CLIP (Contrastive Language–Image Pretraining) model.

## Why this project is worth putting on your CV

Most student CV projects say "trained a CNN to classify cats vs dogs."
This one shows you understand **multimodal embeddings** — the same idea behind
Google Image search-by-text, Pinterest visual search, and CLIP-guided
generative models (Stable Diffusion, DALL·E). It's CV + NLP in one project,
and it's genuinely useful (zero-shot, no labeled dataset needed).

## How it works

1. CLIP has two encoders: one for images, one for text. Both are trained so
   that matching image/caption pairs land close together in the same
   512-dimensional vector space.
2. We embed every uploaded image once (cached).
3. When you type a query, we embed the text with the *same* space.
4. We rank images by cosine similarity to the query vector and show the
   top matches.

No categories are hard-coded — you could type "a cozy reading nook" or
"someone laughing outdoors" and it will still work, as long as the concept
is visually present in an image.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the CLIP model (~350 MB) from Hugging Face — after
that it's cached locally.

## Deploy it online (free, so you can link it on your CV/LinkedIn)

### Option A: Streamlit Community Cloud (easiest)
1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io → "New app" → pick your repo/branch → set
   main file to `app.py`.
3. Deploy. You'll get a public URL like `yourapp.streamlit.app`.

### Option B: Hugging Face Spaces
1. Create a new Space at https://huggingface.co/new-space → SDK: Streamlit.
2. Upload `app.py` and `requirements.txt` (or push via git).
3. It builds automatically and gives you a public URL.

Either way — put the live link directly on your resume/LinkedIn, not just the
GitHub repo. Recruiters click links they can try in 10 seconds far more than
they read code.

## Ideas to extend it (good talking points in interviews)

- Swap brute-force cosine similarity for **FAISS** to scale to 100k+ images.
- Add **reverse search**: upload an image, find similar images (image→image).
- Fine-tune CLIP on a niche domain (e.g. fashion, medical imagery) — talk
  about why zero-shot models sometimes need domain adaptation.
- Add a confidence threshold and show "no good match" instead of forcing a
  result.

## Suggested CV bullet point

> Built and deployed a zero-shot semantic image search engine using OpenAI's
> CLIP model to jointly embed images and natural-language queries in a shared
> vector space, ranked by cosine similarity; deployed as a live Streamlit web
> app requiring no labeled training data.