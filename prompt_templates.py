"""
Prompt templates for medical report generation
"""

PROMPT_TEMPLATES = {
    "basic": {
        "template": "a chest x-ray showing",
        "description": "Simple completion prompt"
    },
    
    "structured": {
        "template": "Radiology report: Lungs are",
        "description": "Medical report starter"
    },
    
    "clinical": {
        "template": "Clinical findings: Chest X-ray demonstrates",
        "description": "Clinical documentation style"
    },
    
    "pneumonia": {
        "template": "Assessment for pneumonia: The X-ray shows",
        "description": "Pneumonia-focused evaluation"
    }
}