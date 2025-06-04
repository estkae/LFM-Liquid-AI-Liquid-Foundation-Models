import gradio as gr
import torch
from lfm.model import LFModel
from lfm.config import LFM_1B_Config, LFM_3B_Config, LFM_7B_Config
import os
from loguru import logger

# Model configurations
MODEL_CONFIGS = {
    "LFM-1B": LFM_1B_Config,
    "LFM-3B": LFM_3B_Config,
    "LFM-7B": LFM_7B_Config
}

class LFMInterface:
    def __init__(self):
        self.model = None
        self.current_model_name = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_model(self, model_name):
        """Load the selected model"""
        if self.current_model_name == model_name and self.model is not None:
            return f"Model {model_name} already loaded."
        
        try:
            # Clear previous model
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()
            
            # Get configuration
            config = MODEL_CONFIGS[model_name]
            
            # Initialize model
            self.model = LFModel(
                token_dim=config.dim,
                channel_dim=config.dim,
                expert_dim=config.dim,
                adapt_dim=config.dim // 4,
                num_experts=config.n_experts if hasattr(config, 'n_experts') else 4
            )
            
            # Check for saved model
            model_path = f"checkpoints/{model_name.lower()}.pt"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded checkpoint from {model_path}")
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Use half precision for GPU
            if self.device.type == "cuda":
                self.model = self.model.half()
            
            self.current_model_name = model_name
            return f"Successfully loaded {model_name} on {self.device}"
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return f"Error loading model: {str(e)}"
    
    def generate_text(self, prompt, max_length, temperature, top_p, top_k):
        """Generate text based on the prompt"""
        if self.model is None:
            return "Please load a model first."
        
        try:
            # Tokenize input (simplified - in production use proper tokenizer)
            # For demo purposes, we'll create dummy input
            batch_size = 1
            seq_length = min(len(prompt.split()), 128)
            embedding_dim = self.model.token_mixer.token_dim
            
            # Create input tensor
            input_tensor = torch.randn(batch_size, seq_length, embedding_dim).to(self.device)
            if self.device.type == "cuda":
                input_tensor = input_tensor.half()
            
            # Generate output
            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        output = self.model(input_tensor)
                else:
                    output = self.model(input_tensor)
            
            # For demo, return a message about the output shape
            # In production, you would decode the output tensor to text
            response = f"Generated output tensor with shape: {output.shape}\n\n"
            response += f"Input prompt: {prompt}\n"
            response += f"Model: {self.current_model_name}\n"
            response += f"Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}\n\n"
            response += "Note: This is a demo. In production, implement proper tokenization and decoding."
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating text: {str(e)}"
    
    def get_model_info(self):
        """Get information about the current model"""
        if self.model is None:
            return "No model loaded."
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = f"Model: {self.current_model_name}\n"
        info += f"Device: {self.device}\n"
        info += f"Total parameters: {total_params:,}\n"
        info += f"Trainable parameters: {trainable_params:,}\n"
        info += f"Memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if self.device.type == "cuda" else "N/A (CPU)"
        
        return info

# Initialize interface
lfm_interface = LFMInterface()

# Create Gradio interface
with gr.Blocks(title="LFM - Liquid Foundation Models") as demo:
    gr.Markdown("# LFM - Liquid Foundation Models Interface")
    gr.Markdown("Interactive interface for Liquid Foundation Models with adaptive computation.")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CONFIGS.keys()),
                value="LFM-1B",
                label="Select Model"
            )
            load_btn = gr.Button("Load Model", variant="primary")
            model_status = gr.Textbox(label="Model Status", lines=2)
            
            gr.Markdown("### Model Information")
            model_info = gr.Textbox(label="Model Details", lines=5)
            
        with gr.Column(scale=2):
            gr.Markdown("### Text Generation")
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=3
            )
            
            with gr.Row():
                max_length = gr.Slider(
                    minimum=10,
                    maximum=512,
                    value=128,
                    step=1,
                    label="Max Length"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
            
            with gr.Row():
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    label="Top-p"
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top-k"
                )
            
            generate_btn = gr.Button("Generate", variant="primary")
            output_text = gr.Textbox(
                label="Generated Text",
                lines=10,
                interactive=False
            )
    
    # Event handlers
    def load_model_handler(model_name):
        status = lfm_interface.load_model(model_name)
        info = lfm_interface.get_model_info()
        return status, info
    
    load_btn.click(
        fn=load_model_handler,
        inputs=[model_dropdown],
        outputs=[model_status, model_info]
    )
    
    generate_btn.click(
        fn=lfm_interface.generate_text,
        inputs=[prompt_input, max_length, temperature, top_p, top_k],
        outputs=[output_text]
    )
    
    # Load default model on start
    demo.load(
        fn=load_model_handler,
        inputs=[model_dropdown],
        outputs=[model_status, model_info]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )