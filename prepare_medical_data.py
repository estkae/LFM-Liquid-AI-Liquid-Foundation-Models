#!/usr/bin/env python3
"""
Medical Data Preparation Pipeline for LFM Training
Prepares medical datasets for training specialized medical language models
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import logging
from tqdm import tqdm
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDataProcessor:
    """Process and prepare medical datasets for training"""
    
    def __init__(self, tokenizer_name: str = "liquid/lfm-3b"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.medical_abbreviations = self._load_medical_abbreviations()
        self.privacy_patterns = self._create_privacy_patterns()
        
    def _load_medical_abbreviations(self) -> Dict[str, str]:
        """Common medical abbreviations and their expansions"""
        return {
            "pt": "patient",
            "hx": "history",
            "rx": "prescription",
            "dx": "diagnosis",
            "sx": "symptoms",
            "tx": "treatment",
            "prn": "as needed",
            "bid": "twice daily",
            "tid": "three times daily",
            "qid": "four times daily",
            "po": "by mouth",
            "iv": "intravenous",
            "im": "intramuscular",
            "bp": "blood pressure",
            "hr": "heart rate",
            "rr": "respiratory rate",
            "temp": "temperature",
            "wbc": "white blood cell",
            "rbc": "red blood cell",
            "hgb": "hemoglobin",
            "plt": "platelet",
            "na": "sodium",
            "k": "potassium",
            "cl": "chloride",
            "bun": "blood urea nitrogen",
            "cr": "creatinine",
            "ast": "aspartate aminotransferase",
            "alt": "alanine aminotransferase"
        }
    
    def _create_privacy_patterns(self) -> List[re.Pattern]:
        """Patterns for detecting and removing PHI (Protected Health Information)"""
        return [
            # Names (simple pattern - real implementation needs NER)
            re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
            # Dates
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            re.compile(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'),
            # Phone numbers
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            # SSN
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            # Medical Record Numbers (MRN)
            re.compile(r'\bMRN:?\s*\d+\b', re.IGNORECASE),
            # Email addresses
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            # IP addresses
            re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            # Age > 89 (HIPAA requirement)
            re.compile(r'\b(9\d|[1-9]\d{2,}) years? old\b', re.IGNORECASE),
        ]
    
    def anonymize_text(self, text: str) -> str:
        """Remove or replace PHI from medical text"""
        # Replace detected PHI with tokens
        for pattern in self.privacy_patterns:
            if pattern.pattern.lower().contains('name'):
                text = pattern.sub('[NAME]', text)
            elif 'date' in pattern.pattern.lower() or '/' in pattern.pattern:
                text = pattern.sub('[DATE]', text)
            elif 'phone' in pattern.pattern or 'ssn' in pattern.pattern.lower():
                text = pattern.sub('[ID]', text)
            elif 'mrn' in pattern.pattern.lower():
                text = pattern.sub('[MRN]', text)
            elif '@' in pattern.pattern:
                text = pattern.sub('[EMAIL]', text)
            elif 'years old' in pattern.pattern.lower():
                text = pattern.sub('[AGE] years old', text)
            else:
                text = pattern.sub('[REDACTED]', text)
        
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations for better understanding"""
        words = text.split()
        expanded = []
        
        for word in words:
            lower_word = word.lower().strip('.,;:')
            if lower_word in self.medical_abbreviations:
                # Keep original capitalization style
                if word.isupper():
                    expanded.append(self.medical_abbreviations[lower_word].upper())
                elif word[0].isupper():
                    expanded.append(self.medical_abbreviations[lower_word].capitalize())
                else:
                    expanded.append(self.medical_abbreviations[lower_word])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def normalize_medical_text(self, text: str) -> str:
        """Normalize medical text for training"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize units
        text = re.sub(r'(\d+)\s*(mg|g|ml|l|mmol|mcg)', r'\1\2', text, flags=re.IGNORECASE)
        
        # Normalize vital signs format
        text = re.sub(r'BP:?\s*(\d+/\d+)', r'Blood Pressure: \1', text, flags=re.IGNORECASE)
        text = re.sub(r'HR:?\s*(\d+)', r'Heart Rate: \1', text, flags=re.IGNORECASE)
        text = re.sub(r'T:?\s*(\d+\.?\d*)', r'Temperature: \1', text, flags=re.IGNORECASE)
        
        return text
    
    def validate_medical_content(self, text: str) -> bool:
        """Validate that text contains medical content"""
        medical_keywords = [
            'patient', 'diagnosis', 'treatment', 'symptoms', 'medication',
            'surgery', 'disease', 'condition', 'clinical', 'medical',
            'hospital', 'doctor', 'nurse', 'therapy', 'examination'
        ]
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in medical_keywords if keyword in text_lower)
        
        # Require at least 2 medical keywords
        return keyword_count >= 2 and len(text.split()) >= 10
    
    def prepare_clinical_notes(self, notes: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Prepare clinical notes for training"""
        prepared_data = []
        
        for note in tqdm(notes, desc="Processing clinical notes"):
            # Skip if required fields are missing
            if 'text' not in note:
                continue
            
            # Process text
            text = note['text']
            text = self.anonymize_text(text)
            text = self.expand_abbreviations(text)
            text = self.normalize_medical_text(text)
            
            # Validate content
            if not self.validate_medical_content(text):
                continue
            
            # Create training example
            prepared_note = {
                'text': text,
                'category': note.get('category', 'general'),
                'specialty': note.get('specialty', 'unknown'),
                'task_type': 'clinical_note'
            }
            
            prepared_data.append(prepared_note)
        
        return prepared_data
    
    def prepare_medical_qa(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Prepare medical Q&A pairs for training"""
        prepared_data = []
        
        for qa in tqdm(qa_pairs, desc="Processing Q&A pairs"):
            if 'question' not in qa or 'answer' not in qa:
                continue
            
            # Process question and answer
            question = self.anonymize_text(qa['question'])
            answer = self.anonymize_text(qa['answer'])
            
            # Format for training
            text = f"Question: {question}\n\nAnswer: {answer}"
            
            prepared_qa = {
                'text': text,
                'category': qa.get('category', 'medical_qa'),
                'difficulty': qa.get('difficulty', 'medium'),
                'task_type': 'qa'
            }
            
            prepared_data.append(prepared_qa)
        
        return prepared_data
    
    def prepare_drug_information(self, drug_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Prepare drug information for training"""
        prepared_data = []
        
        for drug in tqdm(drug_data, desc="Processing drug information"):
            if 'name' not in drug:
                continue
            
            # Create comprehensive drug description
            sections = []
            
            if 'generic_name' in drug:
                sections.append(f"Generic Name: {drug['generic_name']}")
            
            if 'brand_names' in drug:
                sections.append(f"Brand Names: {', '.join(drug['brand_names'])}")
            
            if 'drug_class' in drug:
                sections.append(f"Drug Class: {drug['drug_class']}")
            
            if 'indications' in drug:
                sections.append(f"Indications: {drug['indications']}")
            
            if 'contraindications' in drug:
                sections.append(f"Contraindications: {drug['contraindications']}")
            
            if 'side_effects' in drug:
                sections.append(f"Common Side Effects: {drug['side_effects']}")
            
            if 'dosage' in drug:
                sections.append(f"Typical Dosage: {drug['dosage']}")
            
            if 'interactions' in drug:
                sections.append(f"Drug Interactions: {drug['interactions']}")
            
            text = f"Drug Information for {drug['name']}:\n\n" + "\n\n".join(sections)
            
            prepared_drug = {
                'text': text,
                'category': 'drug_information',
                'drug_class': drug.get('drug_class', 'unknown'),
                'task_type': 'drug_info'
            }
            
            prepared_data.append(prepared_drug)
        
        return prepared_data
    
    def prepare_medical_guidelines(self, guidelines: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Prepare medical guidelines and protocols"""
        prepared_data = []
        
        for guideline in tqdm(guidelines, desc="Processing guidelines"):
            if 'title' not in guideline or 'content' not in guideline:
                continue
            
            # Format guideline
            text = f"Medical Guideline: {guideline['title']}\n\n{guideline['content']}"
            
            # Add evidence level if available
            if 'evidence_level' in guideline:
                text += f"\n\nEvidence Level: {guideline['evidence_level']}"
            
            # Add recommendations if available
            if 'recommendations' in guideline:
                text += "\n\nKey Recommendations:\n"
                for i, rec in enumerate(guideline['recommendations'], 1):
                    text += f"{i}. {rec}\n"
            
            prepared_guideline = {
                'text': self.anonymize_text(text),
                'category': 'medical_guideline',
                'specialty': guideline.get('specialty', 'general'),
                'task_type': 'guideline'
            }
            
            prepared_data.append(prepared_guideline)
        
        return prepared_data
    
    def create_instruction_pairs(self, medical_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Create instruction-following pairs from medical data"""
        instruction_templates = {
            'clinical_note': [
                "Summarize the following clinical note:",
                "Extract the key findings from this clinical note:",
                "Identify the diagnoses mentioned in this clinical note:",
                "List the treatments prescribed in this note:"
            ],
            'qa': [
                "Answer the following medical question:",
                "Provide a detailed response to this clinical query:",
                "Explain the medical concept in the question:"
            ],
            'drug_info': [
                "Provide information about the following medication:",
                "What are the key details about this drug:",
                "Summarize the important information for this medication:"
            ],
            'guideline': [
                "Summarize the key points from this medical guideline:",
                "What are the main recommendations in this guideline:",
                "Extract the evidence-based practices from this guideline:"
            ]
        }
        
        instruction_data = []
        
        for item in medical_data:
            task_type = item.get('task_type', 'general')
            if task_type in instruction_templates:
                # Randomly select an instruction template
                templates = instruction_templates[task_type]
                instruction = np.random.choice(templates)
                
                # Create instruction-response pair
                instruction_pair = {
                    'instruction': instruction,
                    'input': item['text'],
                    'output': self._generate_response(item['text'], task_type),
                    'category': item.get('category', 'unknown'),
                    'task_type': f"instruction_{task_type}"
                }
                
                instruction_data.append(instruction_pair)
        
        return instruction_data
    
    def _generate_response(self, text: str, task_type: str) -> str:
        """Generate appropriate response based on task type"""
        # This is a placeholder - in practice, you would use a model or
        # have pre-generated responses
        if task_type == 'clinical_note':
            return "Summary: " + text[:200] + "..."
        elif task_type == 'qa':
            return text.split('\n\nAnswer: ')[-1] if '\n\nAnswer: ' in text else text
        else:
            return text
    
    def tokenize_and_chunk(self, texts: List[str], max_length: int = 2048) -> List[Dict[str, List[int]]]:
        """Tokenize and chunk texts for training"""
        tokenized_data = []
        
        for text in tqdm(texts, desc="Tokenizing"):
            # Tokenize
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            tokenized_data.append({
                'input_ids': tokens['input_ids'][0].tolist(),
                'attention_mask': tokens['attention_mask'][0].tolist()
            })
        
        return tokenized_data
    
    def save_dataset(self, data: List[Dict], output_path: str, split_ratios: Dict[str, float] = None):
        """Save processed dataset with train/val/test splits"""
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
        
        # Shuffle data
        np.random.shuffle(data)
        
        # Calculate split indices
        n = len(data)
        train_idx = int(n * split_ratios['train'])
        val_idx = train_idx + int(n * split_ratios['validation'])
        
        # Split data
        splits = {
            'train': data[:train_idx],
            'validation': data[train_idx:val_idx],
            'test': data[val_idx:]
        }
        
        # Save each split
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in splits.items():
            file_path = output_path / f"{split_name}.jsonl"
            with open(file_path, 'w') as f:
                for item in split_data:
                    f.write(json.dumps(item) + '\n')
            
            logger.info(f"Saved {len(split_data)} examples to {file_path}")
    
    def create_dataset_info(self, data: List[Dict], output_path: str):
        """Create dataset information and statistics"""
        stats = {
            'total_examples': len(data),
            'task_types': {},
            'categories': {},
            'avg_length': 0,
            'max_length': 0,
            'min_length': float('inf')
        }
        
        total_length = 0
        
        for item in data:
            # Count task types
            task_type = item.get('task_type', 'unknown')
            stats['task_types'][task_type] = stats['task_types'].get(task_type, 0) + 1
            
            # Count categories
            category = item.get('category', 'unknown')
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Calculate length statistics
            text_length = len(item.get('text', '').split())
            total_length += text_length
            stats['max_length'] = max(stats['max_length'], text_length)
            stats['min_length'] = min(stats['min_length'], text_length)
        
        stats['avg_length'] = total_length / len(data) if data else 0
        
        # Save statistics
        info_path = Path(output_path) / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset statistics saved to {info_path}")
        return stats


def main():
    """Example usage of the medical data processor"""
    processor = MedicalDataProcessor()
    
    # Example: Process clinical notes
    clinical_notes = [
        {
            'text': 'Patient John Doe, 45 years old, presents with chest pain. BP: 140/90, HR: 88. Dx: Angina. Rx: Nitroglycerin prn.',
            'category': 'cardiology',
            'specialty': 'cardiology'
        },
        {
            'text': 'Mary Smith admitted with pneumonia. Started on IV antibiotics. O2 sats 92% on room air.',
            'category': 'pulmonology',
            'specialty': 'internal_medicine'
        }
    ]
    
    # Process clinical notes
    processed_notes = processor.prepare_clinical_notes(clinical_notes)
    print(f"Processed {len(processed_notes)} clinical notes")
    
    # Example: Process Q&A pairs
    qa_pairs = [
        {
            'question': 'What are the symptoms of diabetes?',
            'answer': 'Common symptoms include increased thirst, frequent urination, fatigue, and blurred vision.',
            'category': 'endocrinology'
        }
    ]
    
    processed_qa = processor.prepare_medical_qa(qa_pairs)
    print(f"Processed {len(processed_qa)} Q&A pairs")
    
    # Combine all data
    all_data = processed_notes + processed_qa
    
    # Create instruction pairs
    instruction_data = processor.create_instruction_pairs(all_data)
    
    # Save dataset
    processor.save_dataset(all_data + instruction_data, 'data/medical_dataset')
    
    # Create dataset info
    processor.create_dataset_info(all_data + instruction_data, 'data/medical_dataset')


if __name__ == "__main__":
    main()