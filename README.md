<h1 align="center">🤖 Train Your Digital Twin — Fine-Tune an AI Clone of Yourself</h1>

<p align="center">
  <img src="imgs/readme_header_banner.png" alt="Create the AI CLONE banner" width="75%">
</p>


<p align="center">
  <b> What if your WhatsApp messages could become the training ground for an AI that talks just like you? <br />
  <b> This project fine-tune a large language model on your chat history to create your digital twin.</b>
</p>

<p align="center">
  <a href="https://pytorch.org/get-started/locally/"><img src="https://img.shields.io/badge/PyTorch-2.4+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://github.com/meta-llama/"><img src="https://img.shields.io/badge/Llama 3-Supported-blue?logo=meta" alt="Llama3"></a>
  <a href="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"><img src="https://img.shields.io/badge/Hugging Face-Compatible-yellow?logo=huggingface&logoColor=white" alt="Hugging Face"></a>
  <a href="https://github.com/kinggongzilla/torchtune"><img src="https://img.shields.io/badge/Torchtune-Finetuning-lightgrey?logo=meta" alt="Torchtune"></a>
</p>

## What This Project Does

This repository lets you fine-tune a conversational LLM (Llama 3 or Mistral 7B) on your own WhatsApp conversations.
The goal? To create an AI clone that mirrors your tone, humour, quirks, and personality.

### Core Components

- **Chat preprocessing:** Transform WhatsApp `.txt` exports into structured, model-ready datasets.  
- **QLoRA finetuning:** Efficiently fine-tune massive models on GPUs.  
- **Conversational interface:** Talk to your digital twin directly from the terminal/Gradio.

> 💭 *Think of it as ChatGPT — but it’s you.*

**Supported base models**
- [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)  
- [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

Both models can be fine-tuned using QLoRA adapters, drastically reducing memory usage while preserving performance.

---

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/plefloch123/AICLONE-GPT.git
cd AICLONE-GPT
```

### 2. Install Dependencies

Ensure you have **PyTorch ≥ 2.4** installed ([installation guide](https://pytorch.org/get-started/locally/)).

Then install the **Torchtune** fine-tuning framework:

```bash
git clone https://github.com/plefloch123/torchtune.git
cd torchtune
pip install .
cd ..
```

Note that slight modifications to the torchtune library ChatDataset class code were necessary, hence we're not installing from the official repo. In particular the validate_messages function call is removed, to allow for message threads which are not strictly alternating between human and assistant roles.

## Downloading the base model
### Llama3
Run ```tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir model/llama3 --hf-token <HF_TOKEN>```. Replace <HF_TOKEN> with your hugging face access token. In order to download Llama3 you first need to request access on the Meta Llama3 Huggingface page.

### Mistral 7B Instruct
Run ```tune download mistralai/Mistral-7B-Instruct-v0.2 --output-dir model/mistral```. 

If you downloaded the model in another format (e.g. safetensors), please adjust the checkpoint_files in ```mistral/qlora_train_config.yaml```.

## Obtaining and preprocessing your WhatsApp chats
To prepare your WhatsApp chats for training, follow these steps:

1. Export your WhatsApp chats as .txt files. This can be done directly in the WhatsApp app on your phone, for each chat individually. You can export just one .txt from a single chat or many .txt files from all your chats.
Unfortunately, formatting seems to vary between regions. I am based on Europe, so the regex in the ```preprocess.py``` might have to be adjusted if you are based in a different region.
2. Copy the .txt files you exported into ```data/raw_data```. 
3. Run ```python preprocess.py "YOUR NAME"```. This will convert your raw chats into a sharegpt format suitable for training and saves the JSON files to ```data/preprocessed```. Replace ```YOUR NAME``` with the exact string which represents your name in the exportet WhatsApp .txt files. The script will assign you the "gpt" role and your conversation partners the "user" role.


## Start finetune/training

### Llama3
Run ```tune run lora_finetune_single_device --config config/llama3/qlora_train_config.yaml```

### Mistral
Run ```tune run lora_finetune_single_device --config config/mistral/qlora_train_config.yaml```


## Chatting with your AI clone
### Llama3
Run ```tune run chat.py --config config/llama3/inference_config.yaml```

You can define your own system prompt by changing the ```prompt``` string in the ```config/llama3/inference_config.py``` file.

### Mistral
For mistral to fit onto 24GB I first had to quantize the trained model. 

1. Run ```tune run quantize --config config/mistral/quantization.yaml```
2. Run ```tune run chat.py --config config/mistral/inference_config.yaml```

Running this command loads the finetuned model and let's you have a conversation with it in the commandline.

## Hardware requirements
Approx 16 GB vRAM required for QLoRa Llama3 finetune with 4k context length. I ran the finetune on a RTX 3090.
When experimenting with other models, vRAM requirement might vary.


## FAQ
1. **I trained my model but it did not learn my writing style**
Try training for more than one epoch. You can change this in the ```qlora_train_config.yaml``` file.
2. **The preprocessing script does not work**
You probably need to adjust the regex pattern in ```preprocess.py```. The WhatsApp export format varies from region to region.
3. **I want to train a clone of on group chats**
The current setup does not support group chats. Hence do not export and save them into the ```data/raw_data``` directory. If you do want the model to simulate group chats, I think you have to adjust the preprocessing and ChatDataset of torchtune, such that they support more than 2 roles. I haven't tried this myself.

## Common Issues / Notes
* After training, adjust temperature and top_k parameters in the ```inference_config.yaml``` file. I found a temperature of 0.2 and top_k of 10000 to work well for me. 
* Finetuning works best with English chats. If your chats are in another language, you may need to adjust the preprocessing and training parameters accordingly.
