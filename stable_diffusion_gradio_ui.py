from __future__ import annotations

import gc
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageChops, ImageOps, ImageFilter

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DiffusionPipeline,
)


APP_TITLE = "Local Stable Diffusion Studio"
DEFAULT_MODELS_DIR = os.environ.get("SD_MODELS_DIR", "./models")
DEFAULT_OUTPUT_DIR = Path(os.environ.get("SD_OUTPUT_DIR", "./outputs"))
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------

def pil_to_gallery(images: List[Image.Image]) -> List[Tuple[Image.Image, str]]:
    return [(img, f"{img.width}x{img.height}") for img in images]


def ensure_pil(image: Any) -> Optional[Image.Image]:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGBA")
        return Image.fromarray(arr).convert("RGBA")
    if isinstance(image, str):
        return Image.open(image).convert("RGBA")
    raise TypeError(f"Unsupported image type: {type(image)}")


def maybe_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def seed_everything(seed: int) -> int:
    if seed is None or seed < 0:
        seed = torch.randint(0, 2**31 - 1, (1,)).item()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32 - 1))
    return seed


def make_generators(seed: int, count: int, device: str) -> List[torch.Generator]:
    base = seed_everything(seed)
    return [torch.Generator(device=device).manual_seed(base + i) for i in range(count)]


def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def list_local_models(models_dir: str) -> List[str]:
    root = Path(models_dir)
    if not root.exists():
        return []
    items: List[str] = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            items.append(str(path))
        elif path.suffix.lower() in {".ckpt", ".safetensors"}:
            items.append(str(path))
    return items


def normalize_prompt(text: str) -> str:
    return (text or "").strip()


def snap_to_multiple_of_8(x: int) -> int:
    return max(64, int(round(x / 8) * 8))


def save_images(images: List[Image.Image], prefix: str) -> List[str]:
    paths = []
    for i, img in enumerate(images):
        filename = DEFAULT_OUTPUT_DIR / f"{prefix}_{i:02d}.png"
        img.save(filename)
        paths.append(str(filename))
    return paths


def editor_to_background_and_mask(editor_value: Dict[str, Any]) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    if not editor_value:
        return None, None
    background = ensure_pil(editor_value.get("background"))
    layers = editor_value.get("layers") or []
    if background is None:
        return None, None

    mask = Image.new("L", background.size, 0)
    for layer in layers:
        layer_img = ensure_pil(layer)
        if layer_img is None:
            continue
        alpha = layer_img.getchannel("A")
        mask = ImageChops.lighter(mask, alpha)

    # Fallback if layer alpha is unavailable but composite differs from background.
    if mask.getbbox() is None and editor_value.get("composite") is not None:
        composite = ensure_pil(editor_value.get("composite"))
        if composite is not None:
            diff = ImageChops.difference(background, composite).convert("L")
            mask = diff.point(lambda p: 255 if p > 8 else 0)

    return background, mask


def expand_canvas(image: Image.Image, left: int, right: int, top: int, bottom: int, fill_mode: str) -> Tuple[Image.Image, Image.Image]:
    image = ensure_pil(image)
    if image is None:
        raise ValueError("Please upload an image first.")

    new_w = image.width + left + right
    new_h = image.height + top + bottom
    if new_w <= 0 or new_h <= 0:
        raise ValueError("Expanded canvas size is invalid.")

    if fill_mode == "transparent":
        expanded = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    elif fill_mode == "white":
        expanded = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 255))
    elif fill_mode == "black":
        expanded = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 255))
    else:
        expanded = Image.new("RGBA", (new_w, new_h), (127, 127, 127, 255))

    expanded.paste(image, (left, top), image)

    mask = Image.new("L", (new_w, new_h), 255)
    inner = Image.new("L", image.size, 0)
    mask.paste(inner, (left, top))
    return expanded, mask


def image_from_outpaint_preview(editor_value: Dict[str, Any]) -> Optional[Image.Image]:
    if not editor_value:
        return None
    return ensure_pil(editor_value.get("background") or editor_value.get("composite"))


# -----------------------------
# Model / pipeline management
# -----------------------------

@dataclass
class PipelineBundle:
    text2img: Any
    img2img: Any
    inpaint: Any
    source: str
    device: str
    dtype: torch.dtype


class PipelineManager:
    def __init__(self) -> None:
        self.bundle: Optional[PipelineBundle] = None

    def unload(self) -> str:
        self.bundle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "Unloaded current pipelines from memory."

    def _common_from_source_kwargs(self, device: str, dtype: torch.dtype) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"torch_dtype": dtype}
        return kwargs

    def _single_file_pipeline_classes(self, source: str):
        name = source.lower()
        if "xl" in name or "sdxl" in name:
            from diffusers import (
                StableDiffusionXLImg2ImgPipeline,
                StableDiffusionXLInpaintPipeline,
                StableDiffusionXLPipeline,
            )
            return StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline
        else:
            from diffusers import (
                StableDiffusionImg2ImgPipeline,
                StableDiffusionInpaintPipeline,
                StableDiffusionPipeline,
            )
            return StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline

    def load(self, source: str, use_safetensors: bool = True, offload: bool = True, disable_safety_checker: bool = True) -> str:
        source = (source or "").strip()
        if not source:
            raise ValueError("Please provide a model path or Hugging Face repo id.")

        if self.bundle and self.bundle.source == source:
            return f"Model already loaded: {source}"

        self.unload()
        device = infer_device()
        dtype = infer_dtype(device)
        kwargs = self._common_from_source_kwargs(device, dtype)

        if disable_safety_checker:
            kwargs["safety_checker"] = None

        if source.endswith((".safetensors", ".ckpt")):
            t2i_cls, i2i_cls, inp_cls = self._single_file_pipeline_classes(source)
            text2img = t2i_cls.from_single_file(source, **kwargs)
            img2img = i2i_cls.from_single_file(source, **kwargs)
            inpaint = inp_cls.from_single_file(source, **kwargs)
        else:
            text2img = AutoPipelineForText2Image.from_pretrained(
                source,
                use_safetensors=use_safetensors,
                **kwargs,
            )
            img2img = AutoPipelineForImage2Image.from_pretrained(
                source,
                use_safetensors=use_safetensors,
                **kwargs,
            )
            inpaint = AutoPipelineForInpainting.from_pretrained(
                source,
                use_safetensors=use_safetensors,
                **kwargs,
            )

        for pipe in (text2img, img2img, inpaint):
            pipe.set_progress_bar_config(disable=True)
            try:
                if offload and device == "cuda":
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(device)
            except Exception:
                pipe.to(device)
            try:
                if device == "cuda" and torch.__version__.startswith("2"):
                    pass
                elif device == "cuda":
                    pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.bundle = PipelineBundle(
            text2img=text2img,
            img2img=img2img,
            inpaint=inpaint,
            source=source,
            device=device,
            dtype=dtype,
        )
        return f"Loaded model: {source} on {device} with dtype={dtype}."

    def require(self) -> PipelineBundle:
        if self.bundle is None:
            raise RuntimeError("No model is loaded yet. Load one in the Model Manager tab first.")
        return self.bundle


PIPELINES = PipelineManager()


# -----------------------------
# Generation functions
# -----------------------------

def txt2img_generate(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    images_per_prompt: int,
    seed: int,
) -> Tuple[List[Tuple[Image.Image, str]], str]:
    bundle = PIPELINES.require()
    prompt = normalize_prompt(prompt)
    if not prompt:
        raise gr.Error("Positive prompt is required.")

    width = snap_to_multiple_of_8(width)
    height = snap_to_multiple_of_8(height)
    actual_seed = seed_everything(seed)
    generators = make_generators(actual_seed, images_per_prompt, bundle.device)

    result = bundle.text2img(
        prompt=prompt,
        negative_prompt=normalize_prompt(negative_prompt),
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=images_per_prompt,
        generator=generators if images_per_prompt > 1 else generators[0],
    )
    images = [maybe_rgb(img) for img in result.images]
    save_images(images, f"txt2img_{actual_seed}")
    info = f"Seed: {actual_seed} | Resolution: {width}x{height} | Saved to: {DEFAULT_OUTPUT_DIR.resolve()}"
    return pil_to_gallery(images), info


def img2img_generate(
    image: Any,
    prompt: str,
    negative_prompt: str,
    strength: float,
    steps: int,
    guidance_scale: float,
    images_per_prompt: int,
    seed: int,
) -> Tuple[List[Tuple[Image.Image, str]], str]:
    bundle = PIPELINES.require()
    init_image = ensure_pil(image)
    if init_image is None:
        raise gr.Error("Please upload an input image.")
    prompt = normalize_prompt(prompt)
    if not prompt:
        raise gr.Error("Positive prompt is required.")

    actual_seed = seed_everything(seed)
    generators = make_generators(actual_seed, images_per_prompt, bundle.device)

    result = bundle.img2img(
        prompt=prompt,
        negative_prompt=normalize_prompt(negative_prompt),
        image=maybe_rgb(init_image),
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=images_per_prompt,
        generator=generators if images_per_prompt > 1 else generators[0],
    )
    images = [maybe_rgb(img) for img in result.images]
    save_images(images, f"img2img_{actual_seed}")
    info = f"Seed: {actual_seed} | Strength: {strength:.2f} | Saved to: {DEFAULT_OUTPUT_DIR.resolve()}"
    return pil_to_gallery(images), info


def inpaint_generate(
    editor_value: Dict[str, Any],
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    strength: float,
    images_per_prompt: int,
    seed: int,
) -> Tuple[List[Tuple[Image.Image, str]], Image.Image, str]:
    bundle = PIPELINES.require()
    base, mask = editor_to_background_and_mask(editor_value)
    if base is None:
        raise gr.Error("Please provide an image in the inpainting editor.")
    if mask is None or mask.getbbox() is None:
        raise gr.Error("Please paint a mask over the area you want to regenerate.")
    prompt = normalize_prompt(prompt)
    if not prompt:
        raise gr.Error("Positive prompt is required.")

    actual_seed = seed_everything(seed)
    generators = make_generators(actual_seed, images_per_prompt, bundle.device)
    result = bundle.inpaint(
        prompt=prompt,
        negative_prompt=normalize_prompt(negative_prompt),
        image=maybe_rgb(base),
        mask_image=mask,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=images_per_prompt,
        generator=generators if images_per_prompt > 1 else generators[0],
    )
    images = [maybe_rgb(img) for img in result.images]
    save_images(images, f"inpaint_{actual_seed}")
    info = f"Seed: {actual_seed} | Masked pixels: {np.count_nonzero(np.array(mask))} | Saved to: {DEFAULT_OUTPUT_DIR.resolve()}"
    return pil_to_gallery(images), mask, info


def outpaint_prepare(image: Any, left: int, right: int, top: int, bottom: int, fill_mode: str) -> Dict[str, Any]:
    expanded, mask = expand_canvas(image, left, right, top, bottom, fill_mode)
    rgba_mask = Image.merge("RGBA", [mask, mask, mask, mask])
    return {"background": expanded, "layers": [rgba_mask], "composite": expanded}


def outpaint_generate(
    image: Any,
    prompt: str,
    negative_prompt: str,
    left: int,
    right: int,
    top: int,
    bottom: int,
    fill_mode: str,
    steps: int,
    guidance_scale: float,
    strength: float,
    images_per_prompt: int,
    seed: int,
) -> Tuple[Dict[str, Any], List[Tuple[Image.Image, str]], str]:
    editor_value = outpaint_prepare(image, left, right, top, bottom, fill_mode)
    gallery, _, info = inpaint_generate(
        editor_value,
        prompt,
        negative_prompt,
        steps,
        guidance_scale,
        strength,
        images_per_prompt,
        seed,
    )
    return editor_value, gallery, info


# -----------------------------
# Extras
# -----------------------------

def upscale_image(image: Any, scale: int, face_enhance: bool) -> Tuple[Image.Image, str]:
    img = ensure_pil(image)
    if img is None:
        raise gr.Error("Please upload an image first.")

    notes: List[str] = []
    output = img.convert("RGB")

    try:
        from realesrgan import RealESRGAN
        device = torch.device(infer_device())
        model = RealESRGAN(device, scale=scale)
        weights = f"weights/RealESRGAN_x{scale}.pth"
        if not Path(weights).exists():
            raise FileNotFoundError(weights)
        model.load_weights(weights)
        output = model.predict(output)
        notes.append(f"RealESRGAN x{scale} used.")
    except Exception:
        output = output.resize((output.width * scale, output.height * scale), Image.Resampling.LANCZOS)
        notes.append("RealESRGAN not available, used PIL Lanczos resize instead.")

    if face_enhance:
        try:
            from gfpgan import GFPGANer
            model_path = os.environ.get("GFPGAN_MODEL", "weights/GFPGANv1.4.pth")
            restorer = GFPGANer(model_path=model_path, upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)
            _, _, restored = restorer.enhance(np.array(output), has_aligned=False, only_center_face=False, paste_back=True)
            output = Image.fromarray(restored)
            notes.append("GFPGAN face restoration applied.")
        except Exception:
            notes.append("GFPGAN not available, skipped face restoration.")

    output_path = DEFAULT_OUTPUT_DIR / f"extras_upscale_{scale}x.png"
    output.save(output_path)
    return output, " ".join(notes) + f" Saved to: {output_path}"


# -----------------------------
# CLIP interrogator / captioning
# -----------------------------

def interrogate_image(image: Any, mode: str) -> str:
    img = ensure_pil(image)
    if img is None:
        raise gr.Error("Please upload an image first.")
    img_rgb = maybe_rgb(img)

    if mode == "clip-interrogator":
        try:
            from clip_interrogator import Config, Interrogator
            device = infer_device()
            config = Config(device=device)
            ci = Interrogator(config)
            return ci.interrogate(img_rgb)
        except Exception as e:
            # fall through to BLIP fallback with note
            prefix = f"CLIP Interrogator unavailable ({e}). Fallback caption:\n\n"
        else:
            prefix = ""
    else:
        prefix = ""

    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        device = infer_device()
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        if device != "cpu":
            model = model.to(device)
        inputs = processor(images=img_rgb, return_tensors="pt")
        if device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=75)
        text = processor.decode(out[0], skip_special_tokens=True)
        return prefix + text
    except Exception as e:
        raise gr.Error(f"Unable to run interrogator or BLIP captioning: {e}")


# -----------------------------
# Model merge
# -----------------------------

def merge_models(
    model_a: str,
    model_b: str,
    model_c: str,
    alpha: float,
    beta: float,
    output_name: str,
) -> str:
    from safetensors.torch import load_file, save_file

    model_a = (model_a or "").strip()
    model_b = (model_b or "").strip()
    model_c = (model_c or "").strip()
    output_name = (output_name or "merged_model.safetensors").strip()

    if not model_a or not model_b:
        raise gr.Error("Provide at least model A and model B safetensors files.")
    if not model_a.endswith(".safetensors") or not model_b.endswith(".safetensors") or (model_c and not model_c.endswith(".safetensors")):
        raise gr.Error("This merge utility currently supports .safetensors files only.")

    sd_a = load_file(model_a)
    sd_b = load_file(model_b)
    sd_c = load_file(model_c) if model_c else None

    keys = set(sd_a.keys()) & set(sd_b.keys())
    if sd_c:
        keys &= set(sd_c.keys())
    if not keys:
        raise gr.Error("No shared tensor keys were found across the selected checkpoints.")

    total = alpha + beta + (1.0 if sd_c else 0.0)
    wa = alpha / total
    wb = beta / total
    wc = 1.0 / total if sd_c else 0.0

    merged = {}
    for k in keys:
        ta = sd_a[k]
        tb = sd_b[k]
        if ta.shape != tb.shape:
            continue
        if sd_c is not None:
            tc = sd_c[k]
            if tc.shape != ta.shape:
                continue
            merged[k] = (ta * wa + tb * wb + tc * wc).contiguous()
        else:
            merged[k] = (ta * wa + tb * wb).contiguous()

    out_path = DEFAULT_OUTPUT_DIR / output_name
    if out_path.suffix != ".safetensors":
        out_path = out_path.with_suffix(".safetensors")
    save_file(merged, str(out_path))
    return f"Merged {len(merged)} tensors and saved to: {out_path}"


# -----------------------------
# UI helpers
# -----------------------------

def refresh_models_dropdown(models_dir: str):
    choices = list_local_models(models_dir)
    return gr.update(choices=choices)


def load_selected_model(model_source: str, models_dir: str) -> str:
    source = (model_source or "").strip()
    if not source:
        local_models = list_local_models(models_dir)
        if local_models:
            source = local_models[0]
    return PIPELINES.load(source)


def handle_exception(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gr.Error:
            raise
        except Exception as e:
            tb = traceback.format_exc(limit=10)
            raise gr.Error(f"{e}\n\n{tb}")
    return wrapper


# -----------------------------
# Gradio app
# -----------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "A local-first Stable Diffusion interface built with Diffusers + Gradio. "
            "Load a model, generate images, edit with masks, outpaint, upscale, and inspect prompts."
        )

        with gr.Tab("Model Manager"):
            with gr.Row():
                models_dir = gr.Textbox(value=DEFAULT_MODELS_DIR, label="Local models folder")
                refresh_btn = gr.Button("Refresh local models")
            local_models = gr.Dropdown(choices=list_local_models(DEFAULT_MODELS_DIR), label="Detected local models/checkpoints", allow_custom_value=True)
            model_source = gr.Textbox(label="Model source (local path, .ckpt/.safetensors, or Hugging Face repo id)", placeholder="e.g. stabilityai/stable-diffusion-xl-base-1.0 or ./models/sdxl")
            with gr.Row():
                load_btn = gr.Button("Load model", variant="primary")
                unload_btn = gr.Button("Unload model")
            model_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Text to Image"):
            with gr.Row():
                with gr.Column(scale=2):
                    txt_prompt = gr.Textbox(label="Positive prompt", lines=5, placeholder="A cinematic portrait of a fox astronaut, volumetric lighting...")
                    txt_negative = gr.Textbox(label="Negative prompt", lines=4, placeholder="blurry, low quality, deformed, extra fingers...")
                    with gr.Row():
                        txt_width = gr.Slider(256, 2048, value=1024, step=8, label="Width")
                        txt_height = gr.Slider(256, 2048, value=1024, step=8, label="Height")
                    with gr.Row():
                        txt_steps = gr.Slider(1, 100, value=30, step=1, label="Steps")
                        txt_cfg = gr.Slider(1, 20, value=7.5, step=0.1, label="CFG scale")
                    with gr.Row():
                        txt_count = gr.Slider(1, 8, value=1, step=1, label="Images")
                        txt_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    txt_btn = gr.Button("Generate", variant="primary")
                    txt_info = gr.Textbox(label="Run info", interactive=False)
                with gr.Column(scale=3):
                    txt_gallery = gr.Gallery(label="Generated images", columns=2, preview=True, object_fit="contain")

        with gr.Tab("Image to Image"):
            with gr.Row():
                with gr.Column(scale=2):
                    i2i_image = gr.Image(type="pil", label="Input image")
                    i2i_prompt = gr.Textbox(label="Positive prompt", lines=5)
                    i2i_negative = gr.Textbox(label="Negative prompt", lines=4)
                    with gr.Row():
                        i2i_strength = gr.Slider(0.0, 1.0, value=0.45, step=0.01, label="Denoising strength")
                        i2i_steps = gr.Slider(1, 100, value=30, step=1, label="Steps")
                    with gr.Row():
                        i2i_cfg = gr.Slider(1, 20, value=7.5, step=0.1, label="CFG scale")
                        i2i_count = gr.Slider(1, 8, value=1, step=1, label="Images")
                    i2i_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    i2i_btn = gr.Button("Generate", variant="primary")
                    i2i_info = gr.Textbox(label="Run info", interactive=False)
                with gr.Column(scale=3):
                    i2i_gallery = gr.Gallery(label="Results", columns=2, preview=True, object_fit="contain")

        with gr.Tab("Inpainting"):
            with gr.Row():
                with gr.Column(scale=2):
                    inpaint_editor = gr.ImageEditor(type="pil", label="Upload image, then paint over the area to regenerate")
                    inpaint_prompt = gr.Textbox(label="Positive prompt", lines=5)
                    inpaint_negative = gr.Textbox(label="Negative prompt", lines=4)
                    with gr.Row():
                        inpaint_strength = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Strength")
                        inpaint_steps = gr.Slider(1, 100, value=30, step=1, label="Steps")
                    with gr.Row():
                        inpaint_cfg = gr.Slider(1, 20, value=7.5, step=0.1, label="CFG scale")
                        inpaint_count = gr.Slider(1, 8, value=1, step=1, label="Images")
                    inpaint_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    inpaint_btn = gr.Button("Run inpainting", variant="primary")
                    inpaint_info = gr.Textbox(label="Run info", interactive=False)
                with gr.Column(scale=3):
                    inpaint_mask_preview = gr.Image(type="pil", label="Derived mask preview")
                    inpaint_gallery = gr.Gallery(label="Results", columns=2, preview=True, object_fit="contain")

        with gr.Tab("Outpainting"):
            with gr.Row():
                with gr.Column(scale=2):
                    outpaint_image = gr.Image(type="pil", label="Input image")
                    outpaint_prompt = gr.Textbox(label="Positive prompt", lines=5)
                    outpaint_negative = gr.Textbox(label="Negative prompt", lines=4)
                    with gr.Row():
                        out_left = gr.Slider(0, 1024, value=128, step=8, label="Expand left")
                        out_right = gr.Slider(0, 1024, value=128, step=8, label="Expand right")
                    with gr.Row():
                        out_top = gr.Slider(0, 1024, value=128, step=8, label="Expand top")
                        out_bottom = gr.Slider(0, 1024, value=128, step=8, label="Expand bottom")
                    out_fill = gr.Dropdown(["transparent", "white", "black", "gray"], value="transparent", label="Fill for new canvas before generation")
                    with gr.Row():
                        out_strength = gr.Slider(0.0, 1.0, value=0.99, step=0.01, label="Strength")
                        out_steps = gr.Slider(1, 100, value=35, step=1, label="Steps")
                    with gr.Row():
                        out_cfg = gr.Slider(1, 20, value=7.5, step=0.1, label="CFG scale")
                        out_count = gr.Slider(1, 8, value=1, step=1, label="Images")
                    out_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    out_btn = gr.Button("Run outpainting", variant="primary")
                    out_info = gr.Textbox(label="Run info", interactive=False)
                with gr.Column(scale=3):
                    out_preview = gr.ImageEditor(type="pil", label="Expanded canvas + auto mask preview")
                    out_gallery = gr.Gallery(label="Results", columns=2, preview=True, object_fit="contain")

        with gr.Tab("Extras"):
            with gr.Row():
                with gr.Column(scale=2):
                    extras_image = gr.Image(type="pil", label="Image")
                    extras_scale = gr.Dropdown([2, 4], value=2, label="Upscale factor")
                    extras_face = gr.Checkbox(value=True, label="Restore faces when GFPGAN is installed")
                    extras_btn = gr.Button("Upscale / face-fix", variant="primary")
                    extras_info = gr.Textbox(label="Info", interactive=False)
                with gr.Column(scale=3):
                    extras_output = gr.Image(type="pil", label="Output")

        with gr.Tab("CLIP Interrogator"):
            with gr.Row():
                with gr.Column(scale=2):
                    interrogator_image = gr.Image(type="pil", label="Image to analyze")
                    interrogator_mode = gr.Radio(["clip-interrogator", "blip-caption"], value="clip-interrogator", label="Mode")
                    interrogator_btn = gr.Button("Interrogate", variant="primary")
                with gr.Column(scale=3):
                    interrogator_output = gr.Textbox(label="Recovered / guessed prompt", lines=8)

        with gr.Tab("Model Merge"):
            gr.Markdown("Merge compatible `.safetensors` checkpoints with simple weighted arithmetic. Use cautiously and keep backups.")
            with gr.Row():
                merge_a = gr.Textbox(label="Model A (.safetensors path)")
                merge_b = gr.Textbox(label="Model B (.safetensors path)")
                merge_c = gr.Textbox(label="Optional Model C (.safetensors path)")
            with gr.Row():
                merge_alpha = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Weight for A")
                merge_beta = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Weight for B")
            merge_output_name = gr.Textbox(value="merged_model.safetensors", label="Output filename")
            merge_btn = gr.Button("Merge models", variant="primary")
            merge_status = gr.Textbox(label="Status", interactive=False)

        # Events
        refresh_btn.click(refresh_models_dropdown, inputs=models_dir, outputs=local_models)
        local_models.change(lambda x: x, inputs=local_models, outputs=model_source)
        load_btn.click(handle_exception(load_selected_model), inputs=[model_source, models_dir], outputs=model_status)
        unload_btn.click(lambda: PIPELINES.unload(), outputs=model_status)

        txt_btn.click(
            handle_exception(txt2img_generate),
            inputs=[txt_prompt, txt_negative, txt_width, txt_height, txt_steps, txt_cfg, txt_count, txt_seed],
            outputs=[txt_gallery, txt_info],
        )
        i2i_btn.click(
            handle_exception(img2img_generate),
            inputs=[i2i_image, i2i_prompt, i2i_negative, i2i_strength, i2i_steps, i2i_cfg, i2i_count, i2i_seed],
            outputs=[i2i_gallery, i2i_info],
        )
        inpaint_btn.click(
            handle_exception(inpaint_generate),
            inputs=[inpaint_editor, inpaint_prompt, inpaint_negative, inpaint_steps, inpaint_cfg, inpaint_strength, inpaint_count, inpaint_seed],
            outputs=[inpaint_gallery, inpaint_mask_preview, inpaint_info],
        )
        out_btn.click(
            handle_exception(outpaint_generate),
            inputs=[outpaint_image, outpaint_prompt, outpaint_negative, out_left, out_right, out_top, out_bottom, out_fill, out_steps, out_cfg, out_strength, out_count, out_seed],
            outputs=[out_preview, out_gallery, out_info],
        )
        extras_btn.click(
            handle_exception(upscale_image),
            inputs=[extras_image, extras_scale, extras_face],
            outputs=[extras_output, extras_info],
        )
        interrogator_btn.click(
            handle_exception(interrogate_image),
            inputs=[interrogator_image, interrogator_mode],
            outputs=interrogator_output,
        )
        merge_btn.click(
            handle_exception(merge_models),
            inputs=[merge_a, merge_b, merge_c, merge_alpha, merge_beta, merge_output_name],
            outputs=merge_status,
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False,
        footer_links=["gradio", "settings"],
    )


if __name__ == "__main__":
    main()
