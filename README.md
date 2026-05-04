# Toward Automated Cognitive Assessment in Parkinson’s Disease Using Pretrained Language Models
## Overview
This repository contains code and resources for automated extraction of cognitive-process-related narrative categories from first-person reports of individuals with Parkinson’s disease (PD).

The system identifies seven categories reflecting real-world cognition:
- Location  
- Time  
- Sensory  
- Action  
- Thought  
- Emotion  
- Social Interaction  

These categories enable structured analysis of naturalistic narratives, supporting scalable and ecologically valid cognitive assessment.

We evaluate multiple models for benchmarking:
- **Bio_ClinicalBERT** (span-based extraction)  
- **Meta-Llama-3-8B-Instruct** (QLoRA fine-tuned)  
- **GPT-4o mini** (zero- and few-shot prompting)

## Data
Due to privacy and IRB constraints, raw data is not publicly distributed. The dataset consists of de-identified narrative reports describing everyday tasks (e.g., cooking, shopping, social interactions).
The four categories of these tasks include:
- Household Task
- Social Event
- Outdoor Task  
- Event Planning 

## Environment
```bash
git clone https://github.com/BIDS-Xu-Lab/PD_cognition.git
cd PD_cognition
pip install -r requirements.txt
```

## Instruction Tuning
To instruction tune the Llama model from scratch, you will have to specify your huggingface token in train.py under the Llama directory, and run it by:
```bash
### set CUDA_VISIBLE_DEVICES to multiple GPU for multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py
```

## Inference
For Llama inference, you will have to acquire model weights, and then specify the model directory in inference.py under the corresponding directory. You can run it by:
```bash
python inference.py
```

## Citation
```bibtex
@article{khanna2024cognitive,
  title={Toward Automated Cognitive Assessment in Parkinson’s Disease Using Pretrained Language Models},
  author={Khanna, Varada and Bhatt, Nilay and Shin, Ikgyu and Rosso, Mattia and Tinaz, Sule and Ren, Yang and Xu, Hua and Keloth, Vipina K},
  journal={arXiv preprint arXiv:2511.08806},
  year={2025}
}
```

## Model Licenses and Usage

Meta Llama 3 is licensed under the [Meta Llama 3 Community License](https://ai.meta.com/llama/license/). Use of this model requires acceptance of Meta’s terms and conditions. The license may include restrictions on certain types of commercial usage.  

[Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) is based on BERT (Google) and is released under the Apache 2.0 License. This license permits both academic and commercial use, as well as modification and redistribution. Users are required to include proper attribution and retain the license notice when using or distributing the model.   

GPT-4o mini is not open-source and is accessible only via the OpenAI API. Model weights are not publicly available and cannot be redistributed. All usage must comply with OpenAI’s Terms of Use. Clicl here for the [usage policies](https://platform.openai.com/docs/usage-policies)  

## Important Notes
- This repository does not distribute pretrained model weights
- Users must download or access models independently:
  - Llama 3 via Hugging Face (with approval)
  - Bio_ClinicalBERT via Hugging Face
  - GPT models via OpenAI API
