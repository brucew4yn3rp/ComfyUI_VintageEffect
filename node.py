import torch
import numpy as np
from PIL import Image, ImageEnhance
import io


class VintageEffect:
    """
    ComfyUI node that applies vintage/retro effects through JPG compression,
    color grading, film grain, and vignette to emulate old photo aesthetics.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to apply vintage effect to."}),
                "quality": ("INT", {
                    "default": 70, 
                    "min": 0, 
                    "max": 100, 
                    "step": 1,
                    "tooltip": "JPG compression quality. Lower = more artifacts (0-100)"
                }),
                "passes": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of compression passes. More passes = more degradation"
                }),
                "grain_strength": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Amount of film grain. Set to 0 to disable (0-100)"
                }),
                "vignette_strength": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Vignette darkness around edges. Set to 0 to disable (0-100)"
                }),
                "color_grade": (["None", "Warm", "Cool", "Faded", "Sepia"], {
                    "default": "Faded",
                    "tooltip": "Color grading preset"
                }),
                "saturation": ("INT", {
                    "default": 70,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Color saturation (0 = grayscale, 100 = original, 200 = max boost)"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_vintage"
    CATEGORY = "image/effects"
    DESCRIPTION = "Apply vintage/retro effects with JPG compression artifacts, film grain, vignette, and color grading."

    def apply_vintage(self, images, quality=70, passes=1, grain_strength=5, 
                     vignette_strength=0, color_grade="Faded", saturation=70):
        """
        Apply vintage effect to input images.
        
        Args:
            images: Tensor of images in ComfyUI format [B, H, W, C]
            quality: JPG compression quality (0-100)
            passes: Number of compression passes
            grain_strength: Strength of grain effect (0-100, 0 = disabled)
            vignette_strength: Strength of vignette effect (0-100, 0 = disabled)
            color_grade: Color grading preset
            saturation: Color saturation (0-200, 100 = original)
        
        Returns:
            Processed images tensor
        """
        # Convert 0-100 scale inputs to their actual ranges
        quality_actual = max(1, quality)  # Ensure minimum quality of 1
        saturation_actual = saturation / 100.0  # Convert to 0.0-2.0 range
        vignette_actual = vignette_strength / 100.0  # Convert to 0.0-1.0 range
        grain_actual = grain_strength / 2.0  # Convert to 0-50 range (divide by 2)
        
        result_images = []
        
        for image in images:
            # Convert from torch tensor [H, W, C] to numpy array
            img_np = 255.0 * image.cpu().numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            # Convert to RGB if necessary
            if img_pil.mode in ('RGBA', 'LA', 'P'):
                img_pil = img_pil.convert('RGB')
            
            # Apply color grading and saturation
            if color_grade != "None" or saturation != 100:
                img_pil = self._apply_color_effects(img_pil, color_grade, saturation_actual)
            
            # Add film grain before compression
            if grain_strength > 0:
                img_pil = self._add_film_grain(img_pil, grain_actual)
            
            # Apply vignette effect
            if vignette_strength > 0:
                img_pil = self._apply_vignette(img_pil, vignette_actual)
            
            # Apply JPG compression artifacts
            for _ in range(passes):
                buffer = io.BytesIO()
                img_pil.save(buffer, format='JPEG', quality=quality_actual)
                buffer.seek(0)
                img_pil = Image.open(buffer)
            
            # Convert back to torch tensor
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)
            result_images.append(img_tensor)
        
        # Stack all images back into batch
        result_batch = torch.stack(result_images)
        
        return (result_batch,)
    
    def _apply_color_effects(self, img, color_grade, saturation):
        """Apply color grading and saturation adjustments."""
        # Adjust saturation first
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturation)
        
        # Apply color grade preset
        if color_grade == "Warm":
            # Boost reds/yellows, reduce blues (70s/80s photos)
            img_array = np.array(img, dtype=np.float32)
            img_array[:,:,0] = np.clip(img_array[:,:,0] * 1.15, 0, 255)  # Red
            img_array[:,:,1] = np.clip(img_array[:,:,1] * 1.05, 0, 255)  # Green
            img_array[:,:,2] = np.clip(img_array[:,:,2] * 0.85, 0, 255)  # Blue
            img = Image.fromarray(img_array.astype(np.uint8))
        
        elif color_grade == "Cool":
            # Boost blues, reduce reds (vintage cool tone)
            img_array = np.array(img, dtype=np.float32)
            img_array[:,:,0] = np.clip(img_array[:,:,0] * 0.85, 0, 255)  # Red
            img_array[:,:,1] = np.clip(img_array[:,:,1] * 1.0, 0, 255)   # Green
            img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.2, 0, 255)   # Blue
            img = Image.fromarray(img_array.astype(np.uint8))
        
        elif color_grade == "Faded":
            # Reduce contrast and add slight brightness (old faded photo)
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(0.75)
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(1.15)
        
        elif color_grade == "Sepia":
            # Classic sepia tone
            img_array = np.array(img, dtype=np.float32)
            # Sepia matrix transformation
            sepia_r = img_array[:,:,0] * 0.393 + img_array[:,:,1] * 0.769 + img_array[:,:,2] * 0.189
            sepia_g = img_array[:,:,0] * 0.349 + img_array[:,:,1] * 0.686 + img_array[:,:,2] * 0.168
            sepia_b = img_array[:,:,0] * 0.272 + img_array[:,:,1] * 0.534 + img_array[:,:,2] * 0.131
            
            img_array[:,:,0] = np.clip(sepia_r, 0, 255)
            img_array[:,:,1] = np.clip(sepia_g, 0, 255)
            img_array[:,:,2] = np.clip(sepia_b, 0, 255)
            img = Image.fromarray(img_array.astype(np.uint8))
        
        return img
    
    def _add_film_grain(self, img, strength):
        """Add film grain/noise to the image."""
        img_array = np.array(img, dtype=np.float32)
        
        # Generate noise with normal distribution
        noise = np.random.normal(0, strength, img_array.shape)
        
        # Add noise and clip values to valid range
        img_array = np.clip(img_array + noise, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _apply_vignette(self, img, strength):
        """
        Apply vignette effect (darkening around edges).
        
        Args:
            img: PIL Image
            strength: Vignette strength (0.0-1.0)
        """
        img_array = np.array(img, dtype=np.float32)
        height, width = img_array.shape[:2]
        
        # Create radial gradient from center
        center_y, center_x = height / 2, width / 2
        
        # Create coordinate grids
        Y, X = np.ogrid[:height, :width]
        
        # Calculate distance from center, normalized
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_from_center = dist_from_center / max_dist
        
        # Create vignette mask (1.0 at center, darker at edges)
        # Using quadratic falloff for smoother effect
        vignette_mask = 1.0 - (dist_from_center ** 2) * strength
        vignette_mask = np.clip(vignette_mask, 0, 1)
        
        # Expand mask to match image channels
        if len(img_array.shape) == 3:
            vignette_mask = vignette_mask[:, :, np.newaxis]
        
        # Apply vignette
        img_array = img_array * vignette_mask
        img_array = np.clip(img_array, 0, 255)
        
        return Image.fromarray(img_array.astype(np.uint8))