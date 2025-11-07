import gradio as gr
import torch
from omegaconf import DictConfig
from torchtune import config, utils
from torchtune.data._types import Message
from torchtune.data import MistralChatFormat

class ChatRecipe:
    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)
        utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        ckpt_dict = checkpointer.load_checkpoint()
        self._model = self._setup_model(cfg.model, ckpt_dict[utils.MODEL_KEY])
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(self, model_cfg: DictConfig, model_state_dict: dict) -> torch.nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        with self._device:
            model.setup_caches(max_batch_size=1, dtype=self._dtype)

        return model

    @torch.no_grad()
    def generate(self, cfg: DictConfig, chat_history: list, temperature: float):
        if cfg.checkpointer.model_type == "MISTRAL":
            chat_history = MistralChatFormat.format(chat_history)

        tokens, _ = self._tokenizer.tokenize_messages(chat_history, max_seq_len=cfg.max_new_tokens)
        if tokens[-1] == 128001:
            tokens = tokens[:-1]
        model_input = torch.tensor(tokens, dtype=torch.int, device=self._device)
        generated_tokens = utils.generate(
            model=self._model,
            prompt=model_input,
            max_generated_tokens=cfg.max_new_tokens,
            temperature=temperature,  # Dynamic temperature control
            top_k=cfg.top_k,
            stop_tokens=self._tokenizer.stop_tokens,
        )

        decoded_tokens = self._tokenizer.decode(generated_tokens, truncate_at_eos=False)
        return decoded_tokens

    def remove_eot_id(self, s: str):
        eot_id = "<|eot_id|>"
        return s[:-len(eot_id)] if s.endswith(eot_id) else s

    def chat(self, cfg: DictConfig, user_input: str, chat_history: list, temperature: float):
        chat_history.append(Message(role="user", content=user_input))
        output = self.generate(cfg=cfg, chat_history=chat_history, temperature=temperature)  # 🔥 Pass dynamic temperature
        output = output.split('[/INST ')[-1] if cfg.checkpointer.model_type == "MISTRAL" else output.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[-1]
        output = self.remove_eot_id(output)
        chat_history.append(Message(role="assistant", content=output))
        
        # Convert to list of tuples for Gradio
        formatted_history = [(f"🧑 You: {m.content}", f"🤖 AICLONE-GPT: {chat_history[i + 1].content}") for i, m in enumerate(chat_history[:-1]) if m.role == "user"]

        return formatted_history, chat_history


@config.parse
def main(cfg: DictConfig) -> None:
    # Flush memory
    torch.cuda.empty_cache()
    config.log_config(recipe_name="ChatRecipe", cfg=cfg)

    # Initialize the chatbot
    chatbot = ChatRecipe(cfg=cfg)
    chatbot.setup(cfg=cfg)

    chat_history = []

    def chat_interface(user_input, chat_history, temperature):
        chat_history, raw_history = chatbot.chat(cfg, user_input, chat_history, temperature)
        return chat_history, raw_history  # Returns formatted history for Gradio & raw history for tracking

    # Launch Gradio Interface with a Sleek UI
    with gr.Blocks(css="footer {visibility: hidden}") as interface:
        gr.Markdown("""
            <h1 style='text-align: center; color: #2E8B57;'>🤖 AICLONE-GPT: Your AI Chat Companion</h1>
            <p style='text-align: center; font-size: 16px;'>Adjust temperature to control response randomness.</p>
        """)

        with gr.Row():
            with gr.Column():
                user_input = gr.Textbox(
                    placeholder="Type your message here...", 
                    label="Your Message",
                    lines=1,
                    interactive=True
                )
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=2.0, value=cfg.temperature, step=0.1, label="Temperature 🔥"
                )
                with gr.Row():
                    submit_btn = gr.Button("Send Message 🚀", variant="primary")
                    clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
            
            chatbot_ui = gr.Chatbot(height=500)  # Moved to the right side

        # Hidden state to maintain chat history
        chat_state = gr.State([])

        # Functionality
        submit_btn.click(chat_interface, inputs=[user_input, chat_state, temperature_slider], outputs=[chatbot_ui, chat_state])
        user_input.submit(chat_interface, inputs=[user_input, chat_state, temperature_slider], outputs=[chatbot_ui, chat_state])
        clear_btn.click(lambda: ([], []), outputs=[chatbot_ui, chat_state])
    
    interface.launch(share=True)

if __name__ == "__main__":
    import sys
    sys.exit(main())
