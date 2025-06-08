#!/usr/bin/env python3
"""
Simplified Medical Data Preparation for LFM
Works with minimal dependencies
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleMedicalDataProcessor:
    """Simple medical data processor without heavy dependencies"""
    
    def __init__(self):
        self.medical_specialties = [
            'cardiology', 'neurology', 'oncology', 'radiology',
            'pathology', 'pharmacology', 'emergency', 'pediatrics',
            'surgery', 'psychiatry', 'internal_medicine', 'general'
        ]
        
        self.urgency_keywords = {
            'high': ['emergency', 'urgent', 'critical', 'severe', 'acute', 'immediately'],
            'medium': ['moderate', 'significant', 'concerning', 'abnormal'],
            'low': ['mild', 'minor', 'stable', 'routine', 'follow-up']
        }
    
    def detect_specialty(self, text: str) -> str:
        """Detect medical specialty from text"""
        text_lower = text.lower()
        
        # Keyword mapping for specialties
        specialty_keywords = {
            'cardiology': ['heart', 'cardiac', 'ecg', 'ekg', 'chest pain', 'myocardial', 'coronary'],
            'neurology': ['brain', 'neuro', 'seizure', 'stroke', 'headache', 'cognitive'],
            'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'metastasis'],
            'radiology': ['xray', 'x-ray', 'ct scan', 'mri', 'imaging', 'radiograph'],
            'pediatrics': ['child', 'infant', 'pediatric', 'newborn', 'adolescent'],
            'emergency': ['trauma', 'accident', 'acute', 'emergency', 'urgent care'],
            'psychiatry': ['depression', 'anxiety', 'mental health', 'psychiatric', 'mood'],
            'surgery': ['operation', 'surgical', 'incision', 'procedure', 'post-op']
        }
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return specialty
        
        return 'general'
    
    def detect_urgency(self, text: str) -> str:
        """Detect urgency level from text"""
        text_lower = text.lower()
        
        for level, keywords in self.urgency_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return level
        
        return 'medium'
    
    def anonymize_text(self, text: str) -> str:
        """Remove personal information from text"""
        # Remove common name patterns
        text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
        # Remove dates
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[DATE]', text)
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        # Remove SSN patterns
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        return text
    
    def process_text(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """Process a single medical text"""
        # Anonymize
        anonymized = self.anonymize_text(text)
        
        # Detect specialty and urgency
        specialty = metadata.get('specialty') if metadata else None
        if not specialty:
            specialty = self.detect_specialty(text)
        
        urgency = metadata.get('urgency') if metadata else None
        if not urgency:
            urgency = self.detect_urgency(text)
        
        return {
            'text': anonymized,
            'specialty': specialty,
            'urgency': urgency,
            'metadata': metadata or {}
        }
    
    def process_file(self, file_path: Path) -> List[Dict]:
        """Process a file containing medical texts"""
        processed_data = []
        
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            processed_data.append(self.process_text(item))
                        elif isinstance(item, dict) and 'text' in item:
                            processed_data.append(self.process_text(item['text'], item))
                elif isinstance(data, dict) and 'texts' in data:
                    for text in data['texts']:
                        processed_data.append(self.process_text(text))
        
        elif file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by double newlines for separate entries
                texts = content.split('\n\n')
                for text in texts:
                    if text.strip():
                        processed_data.append(self.process_text(text))
        
        elif file_path.suffix == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if isinstance(item, dict) and 'text' in item:
                            processed_data.append(self.process_text(item['text'], item))
        
        return processed_data
    
    def process_directory(self, input_dir: Path) -> List[Dict]:
        """Process all files in a directory"""
        all_data = []
        
        for file_path in input_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.json', '.txt', '.jsonl']:
                logger.info(f"Processing {file_path}")
                try:
                    data = self.process_file(file_path)
                    all_data.extend(data)
                    logger.info(f"  Processed {len(data)} entries")
                except Exception as e:
                    logger.error(f"  Error processing {file_path}: {e}")
        
        return all_data
    
    def save_dataset(self, data: List[Dict], output_file: str):
        """Save processed data to JSONL format"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(data)} entries to {output_path}")
        
        # Save statistics
        stats = {
            'total_entries': len(data),
            'specialties': {},
            'urgency_levels': {}
        }
        
        for item in data:
            specialty = item.get('specialty', 'unknown')
            urgency = item.get('urgency', 'unknown')
            
            stats['specialties'][specialty] = stats['specialties'].get(specialty, 0) + 1
            stats['urgency_levels'][urgency] = stats['urgency_levels'].get(urgency, 0) + 1
        
        stats_file = output_path.with_suffix('.stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")
        return stats


def create_demo_data():
    """Create demo medical data"""
    demo_data = [
        {
            "text": "45-year-old patient presents with severe chest pain radiating to left arm. ECG shows ST elevation in leads II, III, and aVF. Troponin levels elevated. Started on aspirin, nitroglycerin, and heparin. Urgent cardiac catheterization recommended.",
            "specialty": "cardiology",
            "urgency": "high"
        },
        {
            "text": "MRI brain shows 3cm enhancing mass in right temporal lobe with surrounding edema. Differential includes glioblastoma vs metastatic disease. Recommend neurosurgery consultation for biopsy.",
            "specialty": "radiology",
            "urgency": "high"
        },
        {
            "text": "6-year-old with fever 103F for 3 days, barking cough, and inspiratory stridor. Chest X-ray shows steeple sign. Clinical diagnosis of croup. Started on dexamethasone and racemic epinephrine.",
            "specialty": "pediatrics",
            "urgency": "medium"
        },
        {
            "text": "Patient reports worsening depression over past month with anhedonia, insomnia, and passive suicidal ideation. No active plan. Started on sertraline 50mg daily and referred for cognitive behavioral therapy.",
            "specialty": "psychiatry",
            "urgency": "medium"
        },
        {
            "text": "Routine colonoscopy in 52-year-old for screening. Found and removed two small adenomatous polyps. No evidence of malignancy. Recommend repeat colonoscopy in 5 years.",
            "specialty": "gastroenterology",
            "urgency": "low"
        }
    ]
    return demo_data


def main():
    parser = argparse.ArgumentParser(
        description='Prepare medical data for LFM training (simplified version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process demo data
  python prepare_medical_data_simple.py
  
  # Process a directory of medical texts
  python prepare_medical_data_simple.py --input_dir ./medical_data --output_file train_medical.jsonl
  
  # Filter by specialties
  python prepare_medical_data_simple.py --input_dir ./data --specialties cardiology,neurology
        """
    )
    
    parser.add_argument('--input_dir', type=str, help='Directory containing medical data files')
    parser.add_argument('--output_file', type=str, default='train_medical.jsonl',
                       help='Output JSONL file (default: train_medical.jsonl)')
    parser.add_argument('--specialties', type=str,
                       help='Comma-separated list of specialties to include (default: all)')
    parser.add_argument('--min_confidence', type=float, default=0.0,
                       help='Minimum confidence threshold (not used in simple version)')
    parser.add_argument('--demo', action='store_true',
                       help='Run with demo data only')
    
    args = parser.parse_args()
    
    processor = SimpleMedicalDataProcessor()
    
    if args.input_dir and not args.demo:
        # Process actual data
        input_path = Path(args.input_dir)
        if not input_path.exists():
            logger.error(f"Input directory {input_path} does not exist")
            return
        
        logger.info(f"Processing files in {input_path}")
        data = processor.process_directory(input_path)
    else:
        # Use demo data
        logger.info("Running in demo mode with sample medical data")
        data = create_demo_data()
        
        # Process demo data
        processed_data = []
        for item in data:
            processed = processor.process_text(item['text'], item)
            processed_data.append(processed)
        data = processed_data
    
    # Filter by specialties if specified
    if args.specialties:
        specialties = [s.strip() for s in args.specialties.split(',')]
        data = [item for item in data if item.get('specialty') in specialties]
        logger.info(f"Filtered to {len(data)} entries for specialties: {specialties}")
    
    # Save the data
    if data:
        stats = processor.save_dataset(data, args.output_file)
        print("\nDataset Statistics:")
        print(f"Total entries: {stats['total_entries']}")
        print("\nSpecialties:")
        for specialty, count in sorted(stats['specialties'].items()):
            print(f"  {specialty}: {count}")
        print("\nUrgency levels:")
        for urgency, count in sorted(stats['urgency_levels'].items()):
            print(f"  {urgency}: {count}")
    else:
        logger.warning("No data to save")


if __name__ == "__main__":
    main()