"""
BLIP-based medical report generator for chest X-rays
"""

import torch
from PIL import Image
import torchvision.transforms.functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration
from .prompt_templates import PROMPT_TEMPLATES

class MedicalReportGenerator:
    """Medical report generator using BLIP vision-language model"""
    
    def __init__(self, model_id="Salesforce/blip-image-captioning-base"):
        """Initialize the model and processor"""
        print("Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f"Model loaded on {self.device}")
    
    def preprocess_image(self, img_pil, target_size=(384, 384)):
        """
        Preprocess chest X-ray image for VLM input
        """
        # Upscale from 28x28 to larger size
        img = img_pil.resize(target_size, Image.Resampling.BICUBIC)
        
        # Convert to RGB
        img = img.convert("RGB")
        
        # Enhance for medical features
        img = F.adjust_contrast(img, 2.5)
        img = F.adjust_sharpness(img, 2.0)
        
        return img
    
    def generate_report(self, image, prompt_type="structured", max_length=50):
        """
        Generate medical report for a chest X-ray
        """
        prompt = PROMPT_TEMPLATES[prompt_type]["template"]
        
        try:
            # Prepare inputs
            inputs = self.processor(image, prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate report
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=20,
                    num_beams=5,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
            
            # Decode and clean
            report = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt if it appears
            if report.startswith(prompt):
                report = report[len(prompt):].strip()
            
            return report
            
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def analyze_reports(self, reports_dict):
        """
        Analyze quality of generated reports
        """
        pneumonia_keywords = ['pneumonia', 'consolidation', 'infiltrate', 
                             'opacity', 'abnormal', 'infection']
        normal_keywords = ['normal', 'clear', 'unremarkable', 'healthy', 
                          'no abnormalities', 'within normal limits']
        
        results = {}
        for idx, data in reports_dict.items():
            true_label = data['true_label']
            results[idx] = {'true_label': true_label, 'matches': {}}
            
            for ptype, report in data['reports'].items():
                report_lower = report.lower()
                has_pneumonia = any(k in report_lower for k in pneumonia_keywords)
                has_normal = any(k in report_lower for k in normal_keywords)
                
                # Check if matches ground truth
                if true_label == 1 and has_pneumonia:
                    results[idx]['matches'][ptype] = True
                elif true_label == 0 and has_normal and not has_pneumonia:
                    results[idx]['matches'][ptype] = True
                else:
                    results[idx]['matches'][ptype] = False
        
        return results