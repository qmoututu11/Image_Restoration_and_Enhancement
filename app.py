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
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.inference import RestorationPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pipeline
try:
    pipeline = RestorationPipeline()
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
    pipeline = None


def process_image(
    input_image: Image.Image,
    tasks: list,
    denoise_strength: float = 0.5,
    sr_scale: int = 4,
    inpaint_prompt: str = "high quality detailed photo, realistic",
    mask_image: Image.Image = None
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
    if pipeline is None:
        return None, [("Error: Pipeline not initialized", "Error")]
    
    if input_image is None:
        return None, []
    
    try:
        # Convert to RGB if needed
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
        
        # Process image
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
        
        # Wrapper function for processing (using default settings)
        def process_wrapper(img, denoise, sr, colorize, inpaint, mask):
            tasks = get_selected_tasks(denoise, sr, colorize, inpaint)
            if not tasks:
                if img is not None:
                    return [(img, "No tasks selected - please select at least one task")]
                return []
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
                mask_image
            ],
            outputs=[gallery]
        )
        
        # Example images - randomly pick 2 from each task
        import random
        example_paths = []
        
        # Denoising: 2 random images
        denoise_dir = Path("data/pairs/denoise/test/input")
        if denoise_dir.exists():
            denoise_files = [f for f in denoise_dir.glob("*.jpg") if f.exists()]
            if denoise_files:
                selected = random.sample(denoise_files, min(2, len(denoise_files)))
                for f in selected:
                    example_paths.append([str(f)])
        
        # Super-resolution: 2 random images
        sr_dir = Path("data/pairs/sr_x4/test/input")
        if sr_dir.exists():
            sr_files = [f for f in sr_dir.glob("*.jpg") if f.exists()]
            if sr_files:
                selected = random.sample(sr_files, min(2, len(sr_files)))
                for f in selected:
                    example_paths.append([str(f)])
        
        # Colorization: 2 random images
        colorize_dir = Path("data/pairs/colorize/test/input")
        if colorize_dir.exists():
            colorize_files = [f for f in colorize_dir.glob("*.png") if f.exists()]
            if colorize_files:
                selected = random.sample(colorize_files, min(2, len(colorize_files)))
                for f in selected:
                    example_paths.append([str(f)])
        
        # Inpainting: 2 random images
        inpaint_dir = Path("data/pairs/inpaint/test/input")
        if inpaint_dir.exists():
            inpaint_files = [f for f in inpaint_dir.glob("*.jpg") if f.exists()]
            if inpaint_files:
                selected = random.sample(inpaint_files, min(2, len(inpaint_files)))
                for f in selected:
                    example_paths.append([str(f)])
        
        # Create example descriptions
        example_descriptions = []
        for i, example_path in enumerate(example_paths):
            path_str = example_path[0]
            if "denoise" in path_str:
                desc = "Noisy image - try Denoising"
            elif "sr_x4" in path_str or "sr" in path_str:
                desc = "Low-resolution image - try Super-Resolution"
            elif "colorize" in path_str:
                desc = "Grayscale image - try Colorization"
            elif "inpaint" in path_str:
                desc = "Damaged image - try Inpainting (upload mask if available)"
            else:
                desc = "Test image - try any task"
            example_descriptions.append(desc)
        
        if example_paths:
            gr.Markdown("### 📸 Example Images")
            gr.Markdown("""
            **Try these examples:**
            - **Noisy images**: Select "Denoising" to remove noise
            - **Low-resolution images**: Select "Super-Resolution" to upscale
            - **Grayscale images**: Select "Colorization" to add color
            - **Damaged images**: Select "Inpainting" to fill missing parts
            
            You can also combine multiple tasks (e.g., Denoising + Colorization for old photos)
            """)
            
            # Create examples with descriptions
            examples_with_info = []
            for path, desc in zip(example_paths, example_descriptions):
                examples_with_info.append([path[0], desc])
            
            gr.Examples(
                examples=example_paths,
                inputs=[input_image],
                label="Click any example to load it, then select tasks and click 'Restore Image'"
            )
    
    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False  # Set to True to create public link
    )

