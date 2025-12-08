#!/usr/bin/env python3
"""
Gradio UI for Image Restoration and Enhancement
Upload old/damaged images and apply restoration models.
"""

import gradio as gr
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import RestorationPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline - will be reinitialized when model type changes
pipeline = None

def initialize_pipeline(model_type="fine_tuned"):
    """Initialize pipeline with specified model type."""
    global pipeline
    try:
        if model_type == "fine_tuned":
            # Use fine-tuned models (default)
            pipeline = RestorationPipeline()
            logger.info("Pipeline initialized with fine-tuned models")
        else:  # pretrained
            # Use pretrained models only
            config = {
                "denoise": {"fine_tuned_dir": "nonexistent", "pretrained_id": "sd-legacy/stable-diffusion-v1-5"},
                "sr": {"fine_tuned_dir": "nonexistent", "pretrained_id": "sd-legacy/stable-diffusion-v1-5"},
                "colorize": {"fine_tuned_dir": "nonexistent", "pretrained_id": "sd-legacy/stable-diffusion-v1-5"},
                "inpaint": {"fine_tuned_dir": "nonexistent", "pretrained_id": "runwayml/stable-diffusion-inpainting"},
            }
            pipeline = RestorationPipeline(config=config)
            logger.info("Pipeline initialized with pretrained models")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        pipeline = None
        return False

# Initialize with fine-tuned models by default
initialize_pipeline(model_type="fine_tuned")


def process_image(
    input_image: Optional[Image.Image],
    tasks: list,
    denoise_strength: float = 0.5,
    sr_scale: int = 4,
    inpaint_prompt: str = "high quality detailed photo, realistic",
    mask_image: Optional[Image.Image] = None
) -> tuple:
    """
    Process uploaded image through selected restoration tasks.
    
    Args:
        input_image: Uploaded image
        tasks: List of selected tasks
        denoise_strength: Denoising strength (0-1)
        sr_scale: Super-resolution scale (2, 3, or 4)
        inpaint_prompt: Text prompt for inpainting
        mask_image: Optional mask for inpainting
    
    Returns:
        Tuple of (final_image, gallery_list)
    """
    global pipeline
    
    # Check input image first
    if input_image is None:
        return None, []
    
    # Check pipeline initialization
    if pipeline is None:
        return None, [("Error: Pipeline not initialized", "Error")]
    
    try:
        # Convert to RGB if needed
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
        
        # Use local pipeline
        kwargs = {
            "denoise_strength": denoise_strength,
            "sr_scale": sr_scale,
            "inpaint_prompt": inpaint_prompt,
            "mask": mask_image
        }
        
        results = pipeline.process(input_image, tasks, **kwargs)
        
        # Prepare gallery images with original first
        gallery_list = [(results.get("original", input_image), "Original")]
        
        # Track the last intermediate result to avoid duplicating final
        last_intermediate = None
        
        if "denoised" in results:
            gallery_list.append((results["denoised"], "Denoised"))
            last_intermediate = results["denoised"]
        
        if "super_resolved" in results:
            gallery_list.append((results["super_resolved"], f"Super-Resolution (x{sr_scale})"))
            last_intermediate = results["super_resolved"]
        
        if "colorized" in results:
            gallery_list.append((results["colorized"], "Colorized"))
            last_intermediate = results["colorized"]
        
        if "inpainted" in results:
            gallery_list.append((results["inpainted"], "Inpainted"))
            last_intermediate = results["inpainted"]
        
        # Only show final result if it's different from the last intermediate step
        # (i.e., when multiple tasks were applied)
        final_img = results.get("final")
        if final_img is not None:
            # Compare images by checking if they're the same object or have different sizes
            if last_intermediate is None or final_img is not last_intermediate:
                # Check if images are actually different (different size or different object)
                if (last_intermediate is None or 
                    final_img.size != last_intermediate.size or 
                    final_img is not last_intermediate):
                    gallery_list.append((final_img, "Final Result"))
        
        return results["final"], gallery_list
    
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        error_msg = f"Error: {str(e)}"
        return None, [(error_msg, "Error")]


def create_interface():
    """Create and launch Gradio interface."""
    
    with gr.Blocks(title="Image Restoration & Enhancement", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🖼️ Image Restoration & Enhancement System
        
        Upload old, damaged, or low-quality images and apply AI-powered restoration:
        - **Denoising**: Remove noise while preserving details
        - **Super-Resolution**: Enhance image resolution (2x, 3x, or 4x)
        - **Colorization**: Add color to grayscale images
        - **Inpainting**: Fill in missing or damaged parts
        
        Select one or more tasks to apply in sequence.
        """)
        
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300
                )
                
                mask_image = gr.Image(
                    label="Optional: Upload Mask for Inpainting (white = area to inpaint)",
                    type="pil",
                    height=150,
                    visible=False
                )
                
                gr.Markdown("### Model Selection")
                model_type = gr.Radio(
                    choices=["Fine-tuned (Local)", "Pretrained (Hugging Face)"],
                    value="Fine-tuned (Local)",
                    label="Model Type",
                    info="Fine-tuned: Your trained models (recommended). Pretrained: Base Stable Diffusion models."
                )
                
                gr.Markdown("### Task Selection")
                denoise_check = gr.Checkbox(label="Denoising", value=False)
                sr_check = gr.Checkbox(label="Super-Resolution", value=False)
                colorize_check = gr.Checkbox(label="Colorization", value=False)
                inpaint_check = gr.Checkbox(label="Inpainting", value=False)
                
                process_btn = gr.Button("🔄 Restore Image", variant="primary", size="lg")
            
            with gr.Column(scale=2, min_width=600):
                gallery = gr.Gallery(
                    label="Results (Original → Processing Steps → Final)",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    height=700,
                    allow_preview=True
                )
        
        # Helper function to collect selected tasks
        def get_selected_tasks(denoise, sr, colorize, inpaint):
            tasks = []
            if denoise:
                tasks.append("denoise")
            if sr:
                tasks.append("sr")
            if colorize:
                tasks.append("colorize")
            if inpaint:
                tasks.append("inpaint")
            return tasks
        
        # Toggle mask visibility based on inpainting checkbox
        def toggle_mask_visibility(inpaint_selected):
            return gr.update(visible=inpaint_selected)
        
        inpaint_check.change(
            fn=toggle_mask_visibility,
            inputs=[inpaint_check],
            outputs=[mask_image]
        )
        
        model_status = gr.Textbox(
            label="Model Status",
            value="✓ Fine-tuned models loaded",
            interactive=False,
            visible=True
        )
        
        # Handle model type change
        def on_model_type_change(model_type_choice):
            if "Fine-tuned" in model_type_choice:
                model_type_str = "fine_tuned"
            else:  # Pretrained
                model_type_str = "pretrained"
            
            success = initialize_pipeline(model_type=model_type_str)
            if success:
                return gr.update(value=f"✓ Model switched to: {model_type_choice}")
            else:
                return gr.update(value=f"✗ Error switching model. Check logs.")
        
        model_type.change(
            fn=on_model_type_change,
            inputs=[model_type],
            outputs=[model_status]
        )
        
        # Wrapper function for processing (using default settings)
        def process_wrapper(img, denoise, sr, colorize, inpaint, mask, model_choice):
            if pipeline is None:
                error_img = Image.new("RGB", (400, 300), color="red")
                return [(error_img, "Error: Pipeline not initialized. Please check model selection.")]
            
            # Check if image is provided
            if img is None:
                return []
            
            tasks = get_selected_tasks(denoise, sr, colorize, inpaint)
            if not tasks:
                return [(img, "No tasks selected - please select at least one task")]
            
            try:
                # Use default settings: strength=0.5, sr_scale=4, default prompt
                _, gallery_list = process_image(
                    img, 
                    tasks, 
                    denoise_strength=0.5,
                    sr_scale=4,
                    inpaint_prompt="high quality detailed photo, realistic",
                    mask_image=mask
                )
                return gallery_list
            except Exception as e:
                logger.error(f"Error in process_wrapper: {e}", exc_info=True)
                error_img = Image.new("RGB", (400, 300), color="red")
                return [(error_img, f"Error: {str(e)}")]
        
        # Process button click
        process_btn.click(
            fn=process_wrapper,
            inputs=[
                input_image,
                denoise_check,
                sr_check,
                colorize_check,
                inpaint_check,
                mask_image,
                model_type
            ],
            outputs=[gallery]
        )
        
        # Example images from demo directory - organized by task
        demo_images_dir = Path("data/demo/images")
        
        if demo_images_dir.exists():
            gr.Markdown("### 📸 Example Images by Task")
            gr.Markdown("""
            Click any example image below to load it, then select the corresponding task and click 'Restore Image'.
            You can also combine multiple tasks (e.g., Denoising + Colorization for old photos).
            """)
            
            # Organize examples by task
            task_examples = {
                "Denoising": {
                    "files": sorted([f for f in demo_images_dir.glob("denoise*.jpg") if f.exists()]),
                    "description": "Noisy images - Select 'Denoising' task to remove noise",
                    "icon": "🔇"
                },
                "Super-Resolution": {
                    "files": sorted([f for f in demo_images_dir.glob("super-resolution*.jpg") if f.exists()]),
                    "description": "Low-resolution images - Select 'Super-Resolution' task to upscale",
                    "icon": "🔍"
                },
                "Colorization": {
                    "files": sorted([f for f in demo_images_dir.glob("colorize*.png") if f.exists()]),
                    "description": "Grayscale images - Select 'Colorization' task to add color",
                    "icon": "🎨"
                },
                "Inpainting": {
                    "files": sorted([f for f in demo_images_dir.glob("inpaint*.jpg") if f.exists()]),
                    "description": "Damaged images - Select 'Inpainting' task to fill missing parts",
                    "icon": "🔧"
                }
            }
            
            # Create example sections for each task
            all_example_paths = []
            for task_name, task_data in task_examples.items():
                if task_data["files"]:
                    with gr.Group():
                        gr.Markdown(f"#### {task_data['icon']} **{task_name}**")
                        gr.Markdown(f"*{task_data['description']}*")
                        
                        # Create examples for this task
                        task_example_paths = [[str(f)] for f in task_data["files"]]
                        all_example_paths.extend(task_example_paths)
                        
                        gr.Examples(
                            examples=task_example_paths,
                            inputs=[input_image],
                            label=f"{task_name} Examples - Click to load"
                        )
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False  # Set to True to create public link
    )

