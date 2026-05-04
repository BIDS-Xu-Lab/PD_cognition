import os
import re
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
adapters_root = "/path/to/llama/lora/adapters/stored/in/training"
test_csv_path = "path/to/test/csv"
output_root = "path/to/output/dir"

if len(sys.argv) != 2:
    raise SystemExit("Usage: python inference_new.py <epoch_num>")

try:
    epoch_num = int(sys.argv[1])
except ValueError:
    raise SystemExit("Error: <epoch_num> must be an integer, e.g. 1")

if epoch_num < 1 or epoch_num > 10:
    raise SystemExit("Error: <epoch_num> must be between 1 and 10")

# Config
os.environ.pop("ACCELERATE_DISABLE_RICH", None)
os.environ["TRANSFORMERS_VERBOSITY"] = "info"

token = os.environ.get("HF_TOKEN")
if not token:
    raise EnvironmentError("HF_TOKEN is not set in the environment.")

adapter_dir = os.path.join(adapters_root, f"epoch_{epoch_num}")

if not os.path.isdir(adapter_dir):
    raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")


epoch_output_dir = os.path.join(output_root, f"epoch_{epoch_num}")
raw_output_dir = os.path.join(epoch_output_dir, "raw_outputs")
combined_output_csv = os.path.join(epoch_output_dir, "all_outputs.csv")

os.makedirs(epoch_output_dir, exist_ok=True)
os.makedirs(raw_output_dir, exist_ok=True)

# Generation settings
max_input_length = 2048
max_new_tokens = 1024
batch_size = 4   # adjust based on GPU memory
do_sample = False
temperature = 0.0
top_p = 1.0

# Prompt template
template = '''You are an information extraction system. Your task is to identify Cognitive medical entities in the input text.

For each entity, output a version of the text where the entity is wrapped as <entity | LABEL>.
Repeat the sentence for each entity if multiple entities exist. Output all variants as a list, separated by semicolons.

An example for this would be the output:

### Input Text: "Morning Routine: Making breakfast and coffee
He woke up and went to his bathroom to brush his teeth and wash his face."

### Output Text:
[
    Morning Routine:<Making breakfast | ACTION> and coffee
He woke up and went to his bathroom to brush his teeth and wash his face.;

    Morning Routine:<Making breakfast and coffee | ACTION>
He woke up and went to his bathroom to brush his teeth and wash his face.;

    Morning Routine:Making breakfast and coffee
He <woke up | ACTION> and went to his bathroom to brush his teeth and wash his face.;

    Morning Routine:Making breakfast and coffee
He woke up and <went to his bathroom | ACTION> to brush his teeth and wash his face.;

    Morning Routine:Making breakfast and coffee
He woke up and went to <his bathroom | LOCATION> to brush his teeth and wash his face.;

    Morning Routine:Making breakfast and coffee
He woke up and went to his bathroom to <brush his teeth | ACTION> and wash his face.;

    Morning Routine:Making breakfast and coffee
He woke up and went to his bathroom to brush his teeth and <wash his face | ACTION> .
]

Annotation Guidelines:
- Wrap only complete words or phrases; do not annotate partial words.
- Include overlapping or nested entities as separate variants.
- Use the exact text and spacing of the original input.
- Only annotate entities explicitly mentioned in the text.

Use only these labels:
use "SENSORY" is tag to depict perceptions related to vision, touch, sound, smell, taste, and internal bodily sensations and states.
use "TIME" for general time, and the specific tags for time of day, season, month, holiday, or day of the week.
use "LOCATION" to denote the mention of the location.
use "ACTION" to denote the mention of an action done.
use "SOCIAL_INTERACTION" to denote the mention of the interactions between people where the subject is interacting with other
use "EMOTION" to denote the mention of the subjects emotions.
use "THOUGHT" to denote the mention of the subject's thoughts.

Label definitions:

SENSORY: This tag is used to encompass perceptions related to: visual e.g., color, light intensity, form, dimension, quantity, surface characteristics, patterns, motion, and spatial relationships, somatosensory e.g., touch, temperature, pain, pressure, vibration, and awareness of body position and movement, auditory e.g., sounds, voices, environmental noises, olfactory smell, gustatory e.g., taste, flavors, and interoception, i.e., perception of internal bodily sensations and physiological states e.g., heartbeat, breathing, hunger, satiety, thirst, internal temperature, internal physical discomfort, energy levels, muscle tension, aches, and the need for waste relief.
TIME: This category encompasses all references to temporality, including specific points during the day, days of the week, months, seasons, holidays, and special or important dates, capturing any mention or indication of when an event, action, or state occurs within the calendar or daily cycle.
LOCATION: References to places or settings where actions or events occur. It includes specific locations including a room or city, or general descriptions of spatial positioning.
ACTION: Descriptions of physical or mental actions performed by the subject or others, this does not entail mental action or others actions.
SOCIAL_INTERACTION: This category includes references to interactions or exchanges between people. It covers both verbal communication and non-verbal social interactions, such as gestures or expressions.
EMOTION: The subject’s emotional experiences or feelings. It includes both positive and negative emotions.
THOUGHT: This tag is used to denote internal mental processes and reflections that do not necessarily lead to concrete actions or plans. It includes the recall of past events, emotions, or facts, and projection of oneself to the past or future cognitive evaluations or decisions/judgements/comments made; self-reflections on one's thoughts, feelings, or actions; attributing thoughts or feelings to oneself or others or understanding/interpreting thoughts, feelings, or action of others

Strict output rules:
1. Output only the annotations for the input text.
2. Output must be in bracketed plain text list format exactly like the example above.
3. Do not output CSV, JSON, markdown, code fences, explanations, or the original input text by itself.
4. Keep all annotated variants inside one bracketed list.
5. Separate variants with semicolons.
6. Do not annotate partial words.
7. Only annotate entities explicitly present in the text.
8. If there are no entities, output:
[]

### Input Text:
{text}

### Output Text:
'''

# Helpers
def extract_input_text(unprocessed_text: str):
    s = str(unprocessed_text)
    if "### Input Text:" in s:
        txt = s.split("### Input Text:", 1)[1].strip()
        if "### Output Text:" in txt:
            txt = txt.split("### Output Text:", 1)[0].strip()
        return txt
    return s.strip()

def preprocess_text(text: str):
    if text is None:
        return None
    text = str(text).strip()
    return text if text else None

def clean_generated_text(text: str):
    if text is None:
        return ""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()

def extract_bracket_list(text: str):
    text = clean_generated_text(text)
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    return text


def save_raw_output(idx: int, prompt: str, generated_text: str):
    raw_path = os.path.join(raw_output_dir, f"output_{idx}.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write("===== PROMPT =====\n")
        f.write(prompt)
        f.write("\n\n===== GENERATED =====\n")
        f.write(generated_text)

def generate_batch(model, tokenizer, prompts):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_texts = []
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    for i, out_ids in enumerate(outputs):
        gen_ids = out_ids[input_lengths[i]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        generated_texts.append(gen_text)

    return generated_texts

# Load test CSV
df = pd.read_csv(test_csv_path)

if "unprocessed" not in df.columns:
    raise ValueError("Expected column 'unprocessed' in the input CSV.")

# Load tokenizer + model
print(f"Running inference for epoch_{epoch_num}")
print(f"Adapter dir: {adapter_dir}")
print(f"Output dir : {epoch_output_dir}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    token=token,
    use_fast=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    cache_dir="/home/nb752/palmer_scratch/cogn_ex/huggingface_models/",
    token=token,
    device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()

print("Model loaded successfully.")



# Smoke test
smoke_text = (
    "Morning Routine: Making breakfast and coffee\n"
    "He woke up and went to his bathroom to brush his teeth and wash his face."
)

smoke_prompt = template.format(text=smoke_text)
smoke_output = generate_batch(model, tokenizer, [smoke_prompt])[0]
smoke_output_clean = extract_bracket_list(smoke_output)
save_raw_output(-1, smoke_prompt, smoke_output_clean)

print("\n===== SMOKE TEST OUTPUT =====\n")
print(smoke_output_clean)
print("\nSaved smoke test raw output.")

# Full inference
all_rows = []

print(f"\nRunning inference on {len(df)} samples...")

for start in tqdm(range(0, len(df), batch_size), desc=f"epoch_{epoch_num}"):
    batch_df = df.iloc[start:start + batch_size]

    prompts = []
    idxs = []
    original_texts = []

    for row_idx, raw in zip(batch_df.index, batch_df["unprocessed"].tolist()):
        txt = preprocess_text(extract_input_text(raw))
        if not txt:
            continue

        prompt = template.format(text=txt)
        prompts.append(prompt)
        idxs.append(row_idx)
        original_texts.append(txt)

    if not prompts:
        continue

    generated_batch = generate_batch(model, tokenizer, prompts)

    for i, gen in enumerate(generated_batch):
        cleaned = extract_bracket_list(gen)
        save_raw_output(idxs[i], prompts[i], cleaned)

        all_rows.append({
            "idx": idxs[i],
            "text": original_texts[i],
            "generated_output": cleaned,
            "raw_model_output": gen.strip(),
            "epoch": epoch_num,
        })

combined_df = pd.DataFrame(all_rows)
combined_df.to_csv(combined_output_csv, index=False)

print("\nInference complete.")
print(f"Raw per-example outputs saved in: {raw_output_dir}")
print(f"Combined CSV saved to: {combined_output_csv}")