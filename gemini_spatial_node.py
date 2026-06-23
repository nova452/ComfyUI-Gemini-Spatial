import base64
import json
import os

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw
from io import BytesIO

load_dotenv()

class GeminiSpatialNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("STRING", {"default": "google/gemini-2.5-flash", "multiline": False}),
                "task_type": (["2D bounding boxes", "Points"], {"default": "2D bounding boxes"}),
                "target": ("STRING", {"multiline": False, "default": "items"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "reasoning_effort": (["none", "minimal", "low", "medium", "high", "xhigh"], {"default": "none"}),
                "service_tier": (["default", "flex", "priority"], {"default": "default"}),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("annotated_image", "json_output")
    FUNCTION = "analyze"
    CATEGORY = "Gemini/Spatial"

    def analyze(self, image, model, task_type, target, temperature, reasoning_effort, service_tier, api_key):
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required. Pass it to the node or set the OPENROUTER_API_KEY environment variable.")

        # Convert ComfyUI image tensor to base64 string
        # ComfyUI Image is [B, H, W, C] in range [0, 1]
        img_tensor = image[0] # Take first batch
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Format prompt
        if task_type == "2D bounding boxes":
            text_prompt = (
                f"Task: Detect {target}.\n"
                "Return a JSON array of objects.\n"
                "Each object must have:\n"
                '- "box_2d": [ymin, xmin, ymax, xmax] (coordinates 0-1000)\n'
                '- "label": text label\n'
                'Example: [{"box_2d": [100, 200, 300, 400], "label": "example"}]\n'
                "Avoid points. Return ONLY the JSON."
            )
        else:
            text_prompt = (
                f"Task: Point to {target}.\n"
                "Return a JSON array of objects.\n"
                "Each object must have:\n"
                '- "point": [y, x] (coordinates 0-1000)\n'
                '- "label": text label\n'
                'Example: [{"point": [500, 500], "label": "example"}]\n'
                "Return ONLY the JSON."
            )

        # Build messages for OpenRouter (OpenAI-compatible format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ]

        # REST API payload for OpenRouter
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }

        # Add reasoning effort if not disabled
        if reasoning_effort != "none":
            payload["reasoning"] = {"effort": reasoning_effort}

        # Add service tier if not default
        if service_tier != "default":
            payload["service_tier"] = service_tier

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        if response.status_code != 200:
            error_msg = f"API Error: {response.status_code} - {response.text}"
            print(error_msg)
            raise ValueError(error_msg)
        
        result_json = response.json()
        try:
            output_text = result_json["choices"][0]["message"]["content"]
            if output_text.startswith("```json"):
                output_text = output_text.split("```json")[1].split("```")[0].strip()
        except (KeyError, IndexError):
            output_text = json.dumps(result_json)
            
        # Draw on the image
        annotated_img = img.copy()
        try:
            data = json.loads(output_text)
            draw = ImageDraw.Draw(annotated_img)
            w, h = annotated_img.size
            
            for item in data:
                # Handle Bounding Box
                box = item.get("box_2d") or item.get("box_2D") or item.get("box") or item.get("bounding_box") or item.get("bounding_box_2d")
                if box and isinstance(box, list) and len(box) == 4:
                    ymin, xmin, ymax, xmax = box
                    left = (xmin / 1000.0) * w
                    top = (ymin / 1000.0) * h
                    right = (xmax / 1000.0) * w
                    bottom = (ymax / 1000.0) * h
                    draw.rectangle([left, top, right, bottom], outline="red", width=3)
                    
                    label = item.get("label", "")
                    if label:
                        # Draw label background
                        draw.rectangle([left, max(0, top-15), left+(len(label)*6), top], fill="red")
                        draw.text((left+2, max(0, top-14)), label, fill="white")
                        
                # Handle Point
                pt = item.get("point") or item.get("point_2d") or item.get("coordinates")
                if pt and isinstance(pt, list) and len(pt) == 2:
                    py, px = pt
                    y = (py / 1000.0) * h
                    x = (px / 1000.0) * w
                    r = 5
                    draw.ellipse([x-r, y-r, x+r, y+r], fill="blue")
                    
                    label = item.get("label", "")
                    if label:
                        draw.text((x+5, y-5), label, fill="blue")
        except Exception as e:
            print(f"Failed to parse or draw JSON: {e}")
            
        # Convert PIL back to tensor
        out_np = np.array(annotated_img).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_np).unsqueeze(0)
        
        return (out_tensor, output_text)

class GeminiSpatialBBoxNode:
    """Converts Gemini Spatial Understanding 2D bounding boxes into masks and bbox JSON."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "json_output": ("STRING", {"multiline": True, "default": "[]"}),
                "label": ("STRING", {"multiline": False, "default": "head"}),
                "drop_size": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "STRING", "INT")
    RETURN_NAMES = ("mask", "masks", "bboxes_json", "match_count")
    FUNCTION = "to_bboxes"
    CATEGORY = "Gemini/Spatial"

    def to_bboxes(self, image, json_output, label, drop_size):
        # Parse JSON
        try:
            data = json.loads(json_output)
        except json.JSONDecodeError:
            text = json_output.strip()
            if text.startswith("```"):
                text = text.split("```", 2)[-1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            data = json.loads(text)

        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            data = [item for sublist in data for item in sublist]

        # Image dimensions
        img_tensor = image[0]  # [H, W, C]
        h, w = img_tensor.shape[0], img_tensor.shape[1]

        # Labels to match
        target_labels = [l.strip().lower() for l in label.split(",") if l.strip()]
        if not target_labels:
            target_labels = [label.strip().lower()]

        # Collect matching boxes
        boxes = []
        for item in data:
            if not isinstance(item, dict):
                continue
            item_label = item.get("label", "").lower()
            if any(tl in item_label for tl in target_labels):
                box = item.get("box_2d") or item.get("box_2D") or item.get("box") or item.get("bounding_box") or item.get("bounding_box_2d")
                if box and isinstance(box, list) and len(box) == 4:
                    boxes.append((box, item.get("label", "")))

        match_count = len(boxes)
        bboxes_list = []

        # Build mask (union of all matching boxes)
        mask_np = np.zeros((h, w), dtype=np.float32)
        # Build individual masks (one per bounding box)
        individual_masks = []

        # Calculate total image area for drop_size filtering
        total_area = h * w
        drop_threshold = drop_size / 100.0  # Convert percentage to fraction

        for box, box_label in boxes:
            ymin, xmin, ymax, xmax = box
            x1 = max(0, int((xmin / 1000.0) * w))
            y1 = max(0, int((ymin / 1000.0) * h))
            x2 = min(w, int((xmax / 1000.0) * w))
            y2 = min(h, int((ymax / 1000.0) * h))

            # Calculate box area and check against drop_size threshold
            box_area = (x2 - x1) * (y2 - y1)
            box_area_fraction = box_area / total_area if total_area > 0 else 0

            # Skip boxes smaller than drop_size threshold
            if box_area_fraction < drop_threshold:
                continue

            bboxes_list.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": box_label,
                "width": x2 - x1,
                "height": y2 - y1
            })

            if y2 > y1 and x2 > x1:
                mask_np[y1:y2, x1:x2] = 1.0

                # Individual mask for this box
                single_mask_np = np.zeros((h, w), dtype=np.float32)
                single_mask_np[y1:y2, x1:x2] = 1.0
                individual_masks.append(torch.from_numpy(single_mask_np))

        mask = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        # Stack individual masks into a batched tensor [N, H, W]
        if individual_masks:
            masks = torch.stack(individual_masks, dim=0)  # [N, H, W]
        else:
            masks = torch.zeros((0, h, w), dtype=torch.float32)  # empty batch

        bboxes_json = json.dumps(bboxes_list)

        return (mask, masks, bboxes_json, match_count)


class GeminiSpatialCoordsNode:
    """Converts Gemini Spatial Understanding point outputs into coordinate JSON."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "json_output": ("STRING", {"multiline": True, "default": "[]"}),
                "label": ("STRING", {"multiline": False, "default": "center"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("coords_json", "match_count")
    FUNCTION = "to_coords"
    CATEGORY = "Gemini/Spatial"

    def to_coords(self, image, json_output, label):
        # Parse JSON
        try:
            data = json.loads(json_output)
        except json.JSONDecodeError:
            text = json_output.strip()
            if text.startswith("```"):
                text = text.split("```", 2)[-1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            data = json.loads(text)

        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            data = [item for sublist in data for item in sublist]

        # Image dimensions
        img_tensor = image[0]  # [H, W, C]
        h, w = img_tensor.shape[0], img_tensor.shape[1]

        # Labels to match
        target_labels = [l.strip().lower() for l in label.split(",") if l.strip()]
        if not target_labels:
            target_labels = [label.strip().lower()]

        # Collect matching points
        coords = []
        for item in data:
            if not isinstance(item, dict):
                continue
            item_label = item.get("label", "").lower()
            if any(tl in item_label for tl in target_labels):
                pt = item.get("point") or item.get("point_2d") or item.get("coordinates")
                if pt and isinstance(pt, list) and len(pt) == 2:
                    py, px = pt
                    x = int((px / 1000.0) * w)
                    y = int((py / 1000.0) * h)
                    coords.append({"x": x, "y": y, "label": item.get("label", "")})

        match_count = len(coords)
        coords_json = json.dumps(coords)

        return (coords_json, match_count)


class GeminiSpatialBboxToCoordsNode:
    """Converts Gemini Spatial bboxes JSON into center-point coords JSON."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes_json": ("STRING", {"multiline": True, "default": "[]"}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("coords_json", "match_count")
    FUNCTION = "bbox_to_coords"
    CATEGORY = "Gemini/Spatial"

    def bbox_to_coords(self, bboxes_json):
        try:
            bboxes = json.loads(bboxes_json)
        except json.JSONDecodeError:
            bboxes = []

        if not isinstance(bboxes, list):
            bboxes = []

        coords = []
        for item in bboxes:
            if not isinstance(item, dict):
                continue
            x1 = item.get("x1", 0)
            y1 = item.get("y1", 0)
            x2 = item.get("x2", 0)
            y2 = item.get("y2", 0)
            x = (x1 + x2) // 2
            y = (y1 + y2) // 2
            coords.append({"x": x, "y": y, "label": item.get("label", "")})

        match_count = len(coords)
        coords_json = json.dumps(coords)

        return (coords_json, match_count)


NODE_CLASS_MAPPINGS = {
    "GeminiSpatialNode": GeminiSpatialNode,
    "GeminiSpatialBBoxNode": GeminiSpatialBBoxNode,
    "GeminiSpatialCoordsNode": GeminiSpatialCoordsNode,
    "GeminiSpatialBboxToCoordsNode": GeminiSpatialBboxToCoordsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiSpatialNode": "Gemini Spatial Understanding",
    "GeminiSpatialBBoxNode": "Gemini Spatial to Bboxes",
    "GeminiSpatialCoordsNode": "Gemini Spatial to Coords",
    "GeminiSpatialBboxToCoordsNode": "Gemini Bboxes to Coords"
}