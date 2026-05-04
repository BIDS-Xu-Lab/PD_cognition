from datasets import load_dataset
import torch, os
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from utils import find_all_linear_names, print_trainable_parameters


os.environ.pop("ACCELERATE_DISABLE_RICH", None)
os.environ["TRANSFORMERS_VERBOSITY"] = "info"

#set HF_TOKEN when you submit a job via bash
token = os.environ.get("HF_TOKEN")

output_dir = "path/to/output/dir"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

os.environ["WANDB_PROJECT"] = output_dir.split("/")[-1]

train_dataset = load_dataset(
    "csv",
    data_files=["path/to/train.csv"],
    split="train"
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    # device_map="auto",
    cache_dir="path/to/cache/dir",
    token=token,
)

base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=token,
    use_fast=True,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=find_all_linear_names(base_model),
    # target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)
print_trainable_parameters(base_model)

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
8. If there are no entities, output []

### Input Text:
{text}

### Output Text:
{output}'''

def formatting_prompts_func(batch):
    formatted = []
    ups = batch["unprocessed"]
    outs = batch["processed"]

    for u, o in zip(ups, outs):
        if "### Input Text:" in u:
            txt = u.split("### Input Text:", 1)[1].strip()
            if "### Output Text:" in txt:
                txt = txt.split("### Output Text:", 1)[0].strip()
        else:
            txt = u.strip()

        target = o.strip()

        formatted.append(
            template.format(text=txt, output=target) + tokenizer.eos_token
        )

    return {"text": formatted}

train_dataset = train_dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=train_dataset.column_names,
)

training_args = SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.3,
    num_train_epochs=10,
    learning_rate=2e-4,
    bf16=True,
    save_strategy="epoch",
    save_total_limit=10,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    ddp_find_unused_parameters=False,
    eval_strategy="no",
    max_length=2048,
    report_to="none",
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    args=training_args,
)

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        if state.is_world_process_zero and state.log_history:
            last = state.log_history[-1]
            loss = last.get("loss")
            lr = last.get("learning_rate")
            if loss is not None or lr is not None:
                print(f"[step {state.global_step}] loss={loss} lr={lr}")

trainer.add_callback(PrintLossCallback())

trainer.train()
# trainer.train(resume_from_checkpoint=True)

trainer.save_model(output_dir)

final_ckpt = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(final_ckpt)
tokenizer.save_pretrained(final_ckpt)

print("Training Complete! Model saved at:", final_ckpt)