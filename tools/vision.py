"""
Vision Tool - Image understanding and analysis.
Uses vision-capable models (LLaVA, etc.) to analyze images.
"""
import base64
from pathlib import Path
from typing import Optional, Union, List
from dataclasses import dataclass
import tempfile
import os
import logging
import time

logger = logging.getLogger(__name__)

from config import settings


@dataclass
class ImageAnalysis:
    """Result of image analysis"""
    filename: str
    description: str
    extracted_text: Optional[str] = None
    objects_detected: Optional[List[str]] = None
    error: Optional[str] = None


class VisionTool:
    """
    Analyzes images using vision-capable LLMs.
    Supports: PNG, JPG, JPEG, GIF, BMP, WEBP
    """
    
    SUPPORTED_TYPES = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    
    def __init__(self):
        self._ollama_client = None
    
    def is_image(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a supported image"""
        return Path(file_path).suffix.lower() in self.SUPPORTED_TYPES
    
    def prepare_image(
        self, 
        file_path: Optional[Union[str, Path]] = None,
        file_bytes: Optional[bytes] = None,
        filename: str = "image"
    ) -> Optional[str]:
        """
        Prepare image for vision model (base64 encode).
        
        Args:
            file_path: Path to image file
            file_bytes: Raw image bytes
            filename: Name of the file
            
        Returns:
            Base64 encoded image string, or None on error
        """
        try:
            if file_bytes:
                return base64.b64encode(file_bytes).decode('utf-8')
            
            if file_path:
                path = Path(file_path)
                if not path.exists():
                    return None
                with open(path, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            
            return None
            
        except Exception as e:
            logger.exception("Error preparing image")
            return None
    
    def analyze_sync(
        self,
        query: str,
        image_base64: str,
        model: str = "llava:latest",
        filename: str = "image"
    ) -> ImageAnalysis:
        """
        Analyze image using vision model (synchronous).
        
        Args:
            query: What to look for in the image
            image_base64: Base64 encoded image
            model: Vision model to use
            filename: Original filename
            
        Returns:
            ImageAnalysis with description and extracted info
        """
        try:
            import httpx

            base_url = settings.ollama_base_url.rstrip("/")
            url = f"{base_url}/api/generate"
            retries = int(getattr(settings, "ollama_retries", 2) or 0)
            backoff = float(getattr(settings, "ollama_retry_backoff_seconds", 0.5) or 0.0)

            timeout = httpx.Timeout(
                connect=float(getattr(settings, "ollama_connect_timeout_seconds", 5.0)),
                read=120.0,
                write=float(getattr(settings, "ollama_write_timeout_seconds", 30.0)),
                pool=5.0,
            )

            response = None
            for attempt in range(retries + 1):
                try:
                    with httpx.Client(timeout=timeout) as client:
                        response = client.post(
                            url,
                            json={
                                "model": model,
                                "prompt": query,
                                "images": [image_base64],
                                "stream": False,
                            },
                        )
                    if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                        time.sleep(backoff * (2**attempt))
                        continue
                    break
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
                    if attempt < retries:
                        time.sleep(backoff * (2**attempt))
                        continue
                    raise

            if response is None:
                raise RuntimeError("No response from vision request")
                
                if response.status_code != 200:
                    return ImageAnalysis(
                        filename=filename,
                        description="",
                        error=f"Vision model error: {response.status_code}"
                    )
                
                result = response.json()
                description = result.get("response", "")
                
                return ImageAnalysis(
                    filename=filename,
                    description=description
                )
                
        except Exception as e:
            return ImageAnalysis(
                filename=filename,
                description="",
                error=f"Vision analysis error: {str(e)}"
            )
    
    async def analyze(
        self,
        query: str,
        image_base64: str,
        model: str = "llava:latest",
        filename: str = "image"
    ) -> ImageAnalysis:
        """Async wrapper - calls sync version"""
        return self.analyze_sync(query, image_base64, model, filename)
    
    async def describe_image(
        self,
        image_base64: str,
        model: str = "llava:latest",
        filename: str = "image"
    ) -> ImageAnalysis:
        """Get a general description of the image"""
        prompt = """Describe this image in detail. Include:
1. Main subjects or objects
2. Colors and composition
3. Any text visible in the image
4. Context or setting
5. Any notable details

Be thorough but concise."""
        
        return await self.analyze(prompt, image_base64, model, filename)
    
    async def extract_text(
        self,
        image_base64: str,
        model: str = "llava:latest",
        filename: str = "image"
    ) -> ImageAnalysis:
        """Extract text from image (OCR-like)"""
        prompt = """Extract all text visible in this image. 
Transcribe exactly what you see, maintaining the original formatting as much as possible.
If there are multiple text elements, separate them clearly.
If no text is visible, say 'No text detected'."""
        
        result = await self.analyze(prompt, image_base64, model, filename)
        if not result.error:
            result.extracted_text = result.description
        return result
    
    async def answer_question(
        self,
        question: str,
        image_base64: str,
        model: str = "llava:latest",
        filename: str = "image"
    ) -> ImageAnalysis:
        """Answer a specific question about the image"""
        prompt = f"""Look at this image and answer the following question:

Question: {question}

Provide a clear and accurate answer based on what you see in the image."""
        
        return await self.analyze(prompt, image_base64, model, filename)
    
    def format_analysis(self, analysis: ImageAnalysis) -> str:
        """Format image analysis for context"""
        if analysis.error:
            return f"[Image: {analysis.filename}] Error: {analysis.error}"
        
        parts = [f"[Image Analysis: {analysis.filename}]"]
        parts.append(analysis.description)
        
        if analysis.extracted_text:
            parts.append(f"\nExtracted Text:\n{analysis.extracted_text}")
        
        if analysis.objects_detected:
            parts.append(f"\nObjects Detected: {', '.join(analysis.objects_detected)}")
        
        return "\n".join(parts)
