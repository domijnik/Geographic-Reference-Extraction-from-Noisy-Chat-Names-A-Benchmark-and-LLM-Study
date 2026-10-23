"""
Run N prompts x M models over a list of channel names via the Hugging Face
Inference Providers router (one OpenAI-compatible endpoint, one token).

pip install openai pandas tqdm
export HF_TOKEN=hf_...          # token with "Make calls to Inference Providers"
python run_experiment.py names.csv   # CSV with a column "name"
# prompts are read from prompt1.txt, prompt2.txt, prompt3.txt (see PROMPT_FILES)
"""
import asyncio, json, os, sys, hashlib
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

client = AsyncOpenAI(base_url="https://router.huggingface.co/v1",
                     api_key=os.environ["HF_TOKEN"])

# Pin a provider with ":provider" so the backend doesn't change mid-run
# (different providers can use different quantizations). Check each model's
# page on huggingface.co for which providers currently serve it.
MODELS = {
    # "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",                       # add :provider
    "apertus-70b":   "swiss-ai/Apertus-70B-Instruct-2509:publicai",
    # "gpt-oss-120b":  "openai/gpt-oss-120b:cerebras",
    # "gpt-oss-20b":   "openai/gpt-oss-20b:groq",
}

# Each prompt lives in its own text file and must contain the placeholder
# {name} where the channel name(s) go. Write literal braces (e.g. in a JSON
# example) as {{ and }}. The dict key is used as the prompt label in results.
PROMPT_FILES = {
    "p1": "prompt-zero-shot.txt",
    "p2": "prompt-few-shot.txt",
    "p3": "prompt-translated.txt",
}


def load_prompts(files):
    prompts = {}
    for label, path in files.items():
        with open(path, encoding="utf-8") as fh:
            text = fh.read().strip()
        if "{name}" not in text:
            sys.exit(f"{path}: missing the {{name}} placeholder")
        try:
            text.format(name="test")
        except (KeyError, IndexError, ValueError) as e:
            sys.exit(f"{path}: stray braces ({e}); write literal braces as {{{{ and }}}}")
        prompts[label] = text
    return prompts


PROMPTS = load_prompts(PROMPT_FILES)

BATCH = 25          # names per request; set to 1 for one-call-per-name
CONCURRENCY = 32    # parallel requests per model; lower if you get 429s
OUT = "results.jsonl"


def key(model, prompt, chunk):
    return hashlib.md5(f"{model}|{prompt}|{'|'.join(chunk)}".encode()).hexdigest()


async def call(sem, model_id, prompt_tmpl, chunk, model_key, prompt_key, done, f):
    k = key(model_key, prompt_key, chunk)
    if k in done:
        return
    text = prompt_tmpl.format(name="\n".join(f"{i+1}. {n}" for i, n in enumerate(chunk))
                              if len(chunk) > 1 else chunk[0])
    extra = {}
    if model_key.startswith("gpt-oss"):
        extra["reasoning_effort"] = "low"          # keeps latency/tokens down
    async with sem:
        for attempt in range(6):
            try:
                r = await client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": text}],
                    temperature=0, max_tokens=4096, **extra)
                out = r.choices[0].message.content
                break
            except Exception as e:
                if attempt == 5:
                    out = f"__ERROR__ {e}"
                await asyncio.sleep(2 ** attempt)
    f.write(json.dumps({"key": k, "model": model_key, "prompt": prompt_key,
                        "names": chunk, "output": out}) + "\n")
    f.flush()


async def main(path):
    names = pd.read_csv(path)["name"].astype(str).tolist()
    chunks = [names[i:i + BATCH] for i in range(0, len(names), BATCH)]
    done = set()
    if os.path.exists(OUT):                         # resume after a crash
        done = {json.loads(l)["key"] for l in open(OUT)
                if "__ERROR__" not in l}
    with open(OUT, "a") as f:
        tasks = []
        for mk, mid in MODELS.items():
            sem = asyncio.Semaphore(CONCURRENCY)    # per-model rate limiting
            for pk, pt in PROMPTS.items():
                for c in chunks:
                    tasks.append(call(sem, mid, pt, c, mk, pk, done, f))
        await tqdm.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))