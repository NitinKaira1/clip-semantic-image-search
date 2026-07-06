# 🔍 Semantic Image Search Engine (CLIP + Streamlit)

Search a set of images using natural language — no labels, no training,
no keyword tags. Powered by OpenAI's **CLIP** (Contrastive Language–Image
Pretraining) model.

Type something like *"a red car on a street"* or *"someone laughing
outdoors"* and the app ranks your uploaded images by how well they match —
purely from visual understanding, with zero manual tagging.

---

## How it works

CLIP has two encoders — one for images, one for text — trained together so
that matching image/caption pairs land close together in the same
512-dimensional vector space.

1. Every uploaded image is converted into a 512-number vector.
2. Your text query is converted into a vector the same way.
3. Images are ranked by **cosine similarity** to the query vector.
4. Top matches are displayed.

No categories are hard-coded. It works because CLIP was pretrained on
~400 million (image, caption) pairs, so it already understands a huge range
of visual concepts out of the box (this is called **zero-shot** learning).

---

## Tech stack

- **Python**
- **PyTorch** — underlying tensor/model math
- **Hugging Face Transformers** — loads and runs the CLIP model
- **Streamlit** — web UI
- **Pillow** — image loading

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/NitinKaira1/clip-semantic-image-search.git
cd clip-semantic-image-search
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 3. Get the CLIP model

The app needs the `openai/clip-vit-base-patch32` model. There are two ways
to get it — try Option A first.

**Option A — Automatic (recommended)**

Just run the app (see Step 4 below). On first run, `transformers` will
automatically download the model (~600MB) from Hugging Face. No action
needed on your part.

**Option B — Manual download (if Option A fails with an SSL error)**

Some networks (corporate/college firewalls, antivirus software doing SSL
inspection) block Python's certificate verification, even though your
browser works fine. If you hit an error like
`SSL: CERTIFICATE_VERIFY_FAILED`, download the model manually instead:

1. Create a folder named `clip-local` inside the project directory.
2. Download each of these files and place them directly inside
   `clip-local/` (right-click each link → "Save link as..." so it
   downloads instead of opening in the browser):

   - [config.json](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json)
   - [preprocessor_config.json](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json)
   - [tokenizer_config.json](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer_config.json)
   - [vocab.json](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json)
   - [merges.txt](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt)
   - [special_tokens_map.json](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/special_tokens_map.json)
   - [pytorch_model.bin](https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin) *(~600MB — the actual model weights)*

3. That's it. `app.py` automatically detects the `clip-local` folder and
   uses it instead of downloading — no code changes needed.

   *(Alternative for Option B: run these in PowerShell instead of clicking
   links one by one — same result, usually more reliable)*

   ```powershell
   mkdir clip-local
   cd clip-local
   Invoke-WebRequest -Uri "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json" -OutFile "config.json"
   Invoke-WebRequest -Uri "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json" -OutFile "preprocessor_config.json"
   Invoke-WebRequest -Uri "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer_config.json" -OutFile "tokenizer_config.json"
   Invoke-WebRequest -Uri "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json" -OutFile "vocab.json"
   Invoke-WebRequest -Uri "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt" -OutFile "merges.txt"
   Invoke-WebRequest -Uri "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/special_tokens_map.json" -OutFile "special_tokens_map.json"
   Invoke-WebRequest -Uri "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin" -OutFile "pytorch_model.bin"
   ```

### 4. Run the app

```bash
streamlit run app.py
```

Your browser should open automatically at `http://localhost:8501`. If not,
open that URL manually.

### 5. Use it

1. Upload 10-30 images in the sidebar (varied subjects work best for a demo).
2. Type a natural-language search query.
3. See the ranked results.

---

## Project structure

```
clip-semantic-image-search/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── README.md
├── .gitignore
└── clip-local/          # (optional, not tracked by git) manually downloaded model files
```

---

## Why this project

Most beginner CV projects are fixed-label classifiers ("cat vs dog"). This
one demonstrates **multimodal embeddings** — the same underlying idea behind
Google's image-search-by-text, Pinterest visual search, and CLIP-guided
generative models like Stable Diffusion. It requires zero labeled training
data and works on any set of images out of the box.

## Ideas to extend

- Swap brute-force cosine similarity for **FAISS** to scale to 100k+ images
- Add reverse image search (upload an image, find similar images)
- Fine-tune CLIP on a niche domain (fashion, medical imaging, etc.)
- Add a similarity-score threshold to show "no good match" instead of
  forcing a low-confidence result

## License

MIT — free to use, modify, and learn from.