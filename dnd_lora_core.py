#!/usr/bin/env python3
"""
D&D SRD LoRA Fine-Tuning Project - Core Library
===============================================

This module contains the core functionality for training, evaluating, and serving
LoRA fine-tuned models on D&D 5e SRD knowledge.

Key Features:
- Model training with LoRA adaptation
- Model comparison and evaluation
- API serving for real-time inference
- Data preparation utilities
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import json
import time
import re
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DnDLoRATrainer:
    """Handles LoRA training for D&D knowledge injection"""
    
    def __init__(self, 
                 model_name: str = "distilgpt2",
                 output_dir: str = "./models/dnd-lora",
                 device: Optional[str] = None):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
    def _load_tokenizer(self):
        """Load and configure tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
        
    def _load_model(self):
        """Load base model for training"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # MPS compatibility
            device_map={"": self.device}
        )
        return model
    
    def setup_lora(self, rank: int = 16, alpha: int = 32) -> None:
        """Configure LoRA adapter"""
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"] if "gpt" in self.model_name.lower() else ["dense"]
        )
        # Type ignore for PEFT model transformation
        self.model = get_peft_model(self.model, lora_config)  # type: ignore
        logger.info(f"LoRA setup complete. Trainable parameters: {self.model.print_trainable_parameters()}")
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Load and prepare training dataset"""
        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        
        def tokenize_function(examples):
            # When batched=True, examples is a dict with lists of values
            texts = [prompt + completion for prompt, completion in 
                    zip(examples["prompt"], examples["completion"])]
            return self.tokenizer(
                texts,
                truncation=True,
                padding=False,  # Let data collator handle padding
                max_length=512
            )
        
        dataset = Dataset.from_list(data)
        return dataset.map(tokenize_function, batched=True)
    
    def train(self, 
              dataset: Dataset,
              num_epochs: int = 3,
              learning_rate: float = 2e-4,
              batch_size: int = 4) -> None:
        """Train the LoRA model"""
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # Disable wandb
            dataloader_num_workers=0,  # MPS compatibility
            fp16=False,  # Disable for MPS
            bf16=False,
        )
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        logger.info(f"Training complete. Model saved to {self.output_dir}")


class DnDModelComparator:
    """Handles comparison between original and LoRA models"""
    
    def __init__(self, 
                 model_name: str = "distilgpt2",
                 lora_path: str = "./models/dnd-lora",
                 device: Optional[str] = None):
        self.model_name = model_name
        self.lora_path = Path(lora_path)
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.original_model = self._load_original_model()
        self.lora_model = self._load_lora_model()
        
    def _load_original_model(self):
        """Load original base model"""
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map={"": self.device}
        )
    
    def _load_lora_model(self):
        """Load LoRA-enhanced model"""
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map={"": self.device}
        )
        return PeftModel.from_pretrained(base_model, str(self.lora_path))
    
    def generate_response(self, 
                         model, 
                         prompt: str, 
                         max_length: int = 100,
                         temperature: float = 0.7) -> str:
        """Generate response from a model"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                max_new_tokens=min(80, max_length - inputs['input_ids'].shape[1]),  # Limit new tokens more strictly
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                top_k=40,  # Reduced for better quality
                top_p=0.85,  # Slightly more restrictive
                repetition_penalty=1.5,  # Increased to prevent repetition
                no_repeat_ngram_size=4,  # Prevent repeating 4-grams
                early_stopping=True,  # Stop when EOS is generated
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_response = response[len(prompt):].strip()
        return self._post_process_response(raw_response)
    
    def count_dnd_terms(self, text: str) -> int:
        """Count D&D-related terms in text"""
        dnd_terms = [
            'STR', 'DEX', 'CON', 'INT', 'WIS', 'CHA',  # Ability scores
            'AC', 'HP', 'hit points', 'damage', 'spell', 'level',  # Core mechanics
            'd4', 'd6', 'd8', 'd10', 'd12', 'd20',  # Dice
            'saving throw', 'challenge rating', 'spell slot',  # Specific mechanics
            'armor class', 'attack roll', 'proficient',  # Combat
            'Elf', 'Kobold', 'Beholder', 'Dragon', 'Tarrasque',  # Creatures
            'Fireball', 'Longsword', 'Greataxe', 'Plate'  # Spells/Equipment
        ]
        
        text_lower = text.lower()
        return sum(1 for term in dnd_terms if term.lower() in text_lower)
    
    def compare_responses(self, prompt: str, **kwargs) -> Dict:
        """Compare responses from both models"""
        original_response = self.generate_response(self.original_model, prompt, **kwargs)
        lora_response = self.generate_response(self.lora_model, prompt, **kwargs)
        
        return {
            "prompt": prompt,
            "original_response": original_response,
            "lora_response": lora_response,
            "original_dnd_terms": self.count_dnd_terms(original_response),
            "lora_dnd_terms": self.count_dnd_terms(lora_response),
            "improvement": self.count_dnd_terms(lora_response) - self.count_dnd_terms(original_response)
        }
    
    def evaluate_on_questions(self, questions: List[str]) -> List[Dict]:
        """Evaluate both models on a list of questions"""
        results = []
        for question in questions:
            result = self.compare_responses(question)
            results.append(result)
            logger.info(f"Evaluated: {question[:50]}... | Improvement: {result['improvement']:+d}")
        return results
    
    def _post_process_response(self, response: str) -> str:
        """Clean up and post-process model response"""
        # Remove common problematic patterns
        response = response.strip()
        
        # Remove repetitive patterns (more than 3 consecutive identical words)
        words = response.split()
        cleaned_words = []
        consecutive_count = 1
        
        for i, word in enumerate(words):
            if i > 0 and word == words[i-1]:
                consecutive_count += 1
                if consecutive_count <= 3:  # Allow up to 3 repetitions
                    cleaned_words.append(word)
            else:
                consecutive_count = 1
                cleaned_words.append(word)
        
        response = " ".join(cleaned_words)
        
        # Handle raw JSON patterns (convert to natural language)
        
        # Pattern for [{'type': 'natural', 'value': X}] -> "X" (handle various quote styles)
        json_pattern = r"\[{'type':\s*['\"]natural['\"]\s*,\s*['\"]value['\"]\s*:\s*(\d+)}\]"
        response = re.sub(json_pattern, r"\1", response)
        
        # More flexible pattern for malformed JSON
        json_pattern2 = r"\[{['\"]type['\"]:\s*['\"]natural['\"],\s*['\"]value['\"]\s*:\s*(\d+)}\]"
        response = re.sub(json_pattern2, r"\1", response)
        
        # Pattern for AC values with JSON
        ac_pattern = r"AC\s*\[{'type':\s*['\"]natural['\"]\s*,\s*['\"]value['\"]\s*:\s*(\d+)}\]"
        response = re.sub(ac_pattern, r"AC \1", response)
        
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 5:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Truncate at common repetitive patterns
        repetitive_patterns = [
            "Challenge Rating 0.25",
            "AC [",
            "The target",
            "When you"
        ]
        
        for pattern in repetitive_patterns:
            if response.count(pattern) > 2:  # If pattern appears more than twice
                # Find the second occurrence and truncate there
                first_pos = response.find(pattern)
                if first_pos != -1:
                    second_pos = response.find(pattern, first_pos + len(pattern))
                    if second_pos != -1:
                        response = response[:second_pos].strip()
                        if not response.endswith('.'):
                            response += '.'
                        break
        
        return response.strip()

class DnDDataProcessor:
    """Handles D&D SRD data processing and preparation"""
    
    @staticmethod
    def load_srd_data(raw_data_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Load D&D SRD data from JSON files (2014 and 2024 editions) from local raw data directory.
        
        Args:
            raw_data_path: Path to raw data directory. If None, defaults to ./data/raw
            
        Returns:
            Dict containing loaded SRD data organized by category
            
        Note:
            This method reads from a local directory structure:
            raw_data_path/
            ├── 2014/
            │   ├── 5e-SRD-Spells.json
            │   ├── 5e-SRD-Monsters.json
            │   └── ... (other SRD files)
            └── 2024/
                ├── 5e-SRD-Skills.json
                └── 5e-SRD-Ability-Scores.json
        """
        # Default to local data/raw directory
        if raw_data_path is None:
            script_dir = Path(__file__).parent
            raw_data_path = script_dir / 'data' / 'raw'
        else:
            raw_data_path = Path(raw_data_path)
        
        data = {}
        
        logger.info(f"Loading SRD data from: {raw_data_path}")
        
        # Load from both 2014 and 2024 directories
        for edition in ['2014', '2024']:
            edition_data = {}
            edition_path = raw_data_path / edition
            
            if not edition_path.exists():
                logger.warning(f"Edition path {edition_path} does not exist, skipping")
                continue
            
            # Get all JSON files directly from the directory (avoiding any subdirectories like test/)
            json_files = list(edition_path.glob('*.json'))
            
            for json_file in json_files:
                filename = json_file.name
                
                # Skip test files or other non-SRD files
                if 'test' in filename.lower() or not filename.startswith('5e-SRD-'):
                    logger.debug(f"Skipping non-SRD file: {filename}")
                    continue
                
                # Create category name from filename
                # Convert "5e-SRD-Ability-Scores.json" -> "ability_scores"
                category = filename.replace('5e-SRD-', '').replace('.json', '').lower().replace('-', '_')
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        
                    # Store data with edition prefix for 2024, regular for 2014
                    key = f"{category}_{edition}" if edition == '2024' else category
                    edition_data[key] = file_data
                    
                    # Handle both list and dict formats
                    count = len(file_data) if isinstance(file_data, list) else len(file_data.keys()) if isinstance(file_data, dict) else 1
                    logger.info(f"Loaded {count} items from {edition}/{filename} -> {key}")
                    
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.error(f"Failed to load {json_file}: {e}")
                    continue
            
            # Merge edition data into main data dict
            data.update(edition_data)
            logger.info(f"Loaded {len(edition_data)} categories from {edition} edition")
        
        logger.info(f"Total categories loaded: {len(data)}")
        return data
    
    @staticmethod
    def create_qa_pairs(srd_data: Dict) -> List[Dict]:
        """Create question-answer pairs from comprehensive SRD data"""
        qa_pairs = []
        
        # Helper function to safely get description
        def get_description(item, desc_keys=['desc', 'description']):
            for key in desc_keys:
                desc = item.get(key, '')
                if isinstance(desc, list) and desc:
                    return desc[0]
                elif isinstance(desc, str) and desc:
                    return desc
            return "No description available."
        
        # Process spells (both editions)
        for spell_key in ['spells', 'spells_2024']:
            for spell in srd_data.get(spell_key, []):
                name = spell.get('name', 'Unknown Spell')
                level = spell.get('level', 0)
                level_text = 'cantrip' if level == 0 else f'level {level}'
                desc = get_description(spell)
                
                qa_pairs.extend([
                    {
                        "prompt": f"What is the {name} spell in D&D?",
                        "completion": f"The {name} spell is a {level_text} spell. {desc}"
                    },
                    {
                        "prompt": f"What level is {name}?",
                        "completion": f"{name} is a {level_text} spell."
                    }
                ])
                
                # Add damage info if available
                if 'damage' in spell:
                    damage_info = spell['damage']
                    damage_dice = None
                    
                    # Try different damage fields
                    if 'damage_at_slot_level' in damage_info:
                        # Get base level damage (usually the spell's level or level 1)
                        slot_levels = damage_info['damage_at_slot_level']
                        if str(level) in slot_levels:
                            damage_dice = slot_levels[str(level)]
                        elif '1' in slot_levels:
                            damage_dice = slot_levels['1']
                        elif slot_levels:
                            # Get first available level
                            damage_dice = next(iter(slot_levels.values()))
                    elif 'damage_dice' in damage_info:
                        damage_dice = damage_info['damage_dice']
                    
                    if damage_dice:
                        qa_pairs.append({
                            "prompt": f"What damage does {name} deal?",
                            "completion": f"{name} deals {damage_dice} damage."
                        })
        
        # Process monsters
        for monster in srd_data.get('monsters', []):
            name = monster.get('name', 'Unknown Monster')
            
            # Parse armor class properly
            ac_raw = monster.get('armor_class', [])
            if isinstance(ac_raw, list) and ac_raw:
                ac_value = ac_raw[0].get('value', 'unknown')
                ac_type = ac_raw[0].get('type', '')
                if ac_type == 'natural':
                    ac = f"{ac_value} (natural armor)"
                elif ac_type == 'dex':
                    ac = f"{ac_value}"
                else:
                    ac = f"{ac_value}"
            else:
                ac = 'unknown'
            
            hp = monster.get('hit_points', 'unknown') 
            cr = monster.get('challenge_rating', 'unknown')
            size = monster.get('size', 'Medium')
            type_name = monster.get('type', 'creature')
            
            qa_pairs.extend([
                {
                    "prompt": f"What is a {name} in D&D?",
                    "completion": f"The {name} is a {size} {type_name} with AC {ac}, {hp} hit points, and Challenge Rating {cr}."
                },
                {
                    "prompt": f"What is the Challenge Rating of a {name}?",
                    "completion": f"The {name} has a Challenge Rating of {cr}."
                },
                {
                    "prompt": f"What is the AC of a {name}?",
                    "completion": f"The {name} has an Armor Class of {ac}."
                }
            ])
        
        # Process equipment
        for item in srd_data.get('equipment', []):
            name = item.get('name', 'Unknown Item')
            desc = get_description(item)
            
            qa_pairs.append({
                "prompt": f"What is a {name} in D&D?",
                "completion": f"The {name} is equipment. {desc}"
            })
            
            # Add weapon-specific info
            if 'weapon_category' in item:
                category = item.get('weapon_category', 'weapon')
                damage_dice = item.get('damage', {}).get('damage_dice', 'varies')
                qa_pairs.extend([
                    {
                        "prompt": f"What damage does a {name} deal?",
                        "completion": f"The {name} is a {category} that deals {damage_dice} damage."
                    },
                    {
                        "prompt": f"What are the stats of a {name}?",
                        "completion": f"The {name} is a {category} that deals {damage_dice} damage."
                    }
                ])
            
            # Add armor-specific info
            if 'armor_category' in item:
                category = item.get('armor_category', 'armor')
                ac_base = item.get('armor_class', {}).get('base', 'varies')
                qa_pairs.append({
                    "prompt": f"What is the AC of {name}?",
                    "completion": f"The {name} is {category} that provides AC {ac_base}."
                })
        
        # Process races (both editions)
        for race_key in ['races', 'races_2024']:
            for race in srd_data.get(race_key, []):
                name = race.get('name', 'Unknown Race')
                ability_increases = race.get('ability_score_increase', {})
                ability_text = ', '.join([f"{k} +{v}" for k, v in ability_increases.items()]) or "varies"
                desc = get_description(race)
                
                qa_pairs.extend([
                    {
                        "prompt": f"What are the racial traits of a {name}?",
                        "completion": f"The {name} has ability score increases: {ability_text}. {desc}"
                    },
                    {
                        "prompt": f"What ability scores does a {name} increase?",
                        "completion": f"The {name} increases: {ability_text}."
                    }
                ])
        
        # Process classes
        for class_info in srd_data.get('classes', []):
            name = class_info.get('name', 'Unknown Class')
            hit_die = class_info.get('hit_die', 6)
            desc = get_description(class_info)
            
            qa_pairs.extend([
                {
                    "prompt": f"What is the {name} class in D&D?",
                    "completion": f"The {name} is a class with hit die d{hit_die}. {desc}"
                },
                {
                    "prompt": f"What hit die does a {name} use?",
                    "completion": f"The {name} class uses a d{hit_die} hit die."
                }
            ])
        
        # Process magic items
        for item in srd_data.get('magic_items', []):
            name = item.get('name', 'Unknown Magic Item')
            desc = get_description(item)
            rarity = item.get('rarity', {}).get('name', 'unknown')
            
            qa_pairs.extend([
                {
                    "prompt": f"What is a {name} in D&D?",
                    "completion": f"The {name} is a {rarity} magic item. {desc}"
                },
                {
                    "prompt": f"What rarity is a {name}?",
                    "completion": f"The {name} is a {rarity} magic item."
                }
            ])
        
        # Process conditions
        for condition in srd_data.get('conditions', []):
            name = condition.get('name', 'Unknown Condition')
            desc = get_description(condition)
            
            qa_pairs.append({
                "prompt": f"What is the {name} condition in D&D?",
                "completion": f"The {name} condition: {desc}"
            })
        
        # Process skills (both editions)
        for skill_key in ['skills', 'skills_2024']:
            for skill in srd_data.get(skill_key, []):
                name = skill.get('name', 'Unknown Skill')
                ability = skill.get('ability_score', {}).get('name', 'unknown')
                desc = get_description(skill)
                
                qa_pairs.extend([
                    {
                        "prompt": f"What is the {name} skill in D&D?",
                        "completion": f"{name} is a skill based on {ability}. {desc}"
                    },
                    {
                        "prompt": f"What ability score is used for {name}?",
                        "completion": f"The {name} skill uses {ability}."
                    }
                ])
        
        # Process feats
        for feat in srd_data.get('feats', []):
            name = feat.get('name', 'Unknown Feat')
            desc = get_description(feat)
            
            qa_pairs.append({
                "prompt": f"What is the {name} feat in D&D?",
                "completion": f"The {name} feat: {desc}"
            })
        
        # Process backgrounds
        for background in srd_data.get('backgrounds', []):
            name = background.get('name', 'Unknown Background')
            desc = get_description(background)
            
            qa_pairs.append({
                "prompt": f"What is the {name} background in D&D?",
                "completion": f"The {name} background: {desc}"
            })
        
        # Process rules and rule sections
        for rule in srd_data.get('rules', []):
            name = rule.get('name', 'Unknown Rule')
            desc = get_description(rule)
            
            qa_pairs.append({
                "prompt": f"What is the rule for {name} in D&D?",
                "completion": f"{name}: {desc}"
            })
        
        for section in srd_data.get('rule_sections', []):
            name = section.get('name', 'Unknown Rule Section')
            desc = get_description(section)
            
            qa_pairs.append({
                "prompt": f"What are the rules for {name} in D&D?",
                "completion": f"{name}: {desc}"
            })
        
        # Process damage types
        for damage_type in srd_data.get('damage_types', []):
            name = damage_type.get('name', 'Unknown Damage Type')
            desc = get_description(damage_type)
            
            qa_pairs.append({
                "prompt": f"What is {name} damage in D&D?",
                "completion": f"{name} damage: {desc}"
            })
        
        # Process languages
        for language in srd_data.get('languages', []):
            name = language.get('name', 'Unknown Language')
            desc = get_description(language)
            
            qa_pairs.append({
                "prompt": f"What is the {name} language in D&D?",
                "completion": f"The {name} language: {desc}"
            })
        
        logger.info(f"Created {len(qa_pairs)} Q&A pairs from {len(srd_data)} data categories")
        return qa_pairs
    
    @staticmethod
    def save_dataset(qa_pairs: List[Dict], output_path: str) -> None:
        """Save Q&A pairs to JSONL format"""
        with open(output_path, 'w') as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair) + '\n')
        logger.info(f"Dataset saved to {output_path}")


# Predefined question sets for evaluation
DND_QUESTIONS = [
    "What is a Fireball spell in D&D?",
    "What are the racial traits of an Elf?", 
    "What is a Beholder in D&D?",
    "What is the Challenge Rating of a Kobold?",
    "What is a saving throw?",
    "What damage does a Greataxe deal?",
    "What is a spell slot in D&D?",
    "What is the Armor Class of Plate Armor?",
    "How do you cast spells in D&D 5e?",
    "What are the stats of a Longsword?"
]

GENERAL_QUESTIONS = [
    "What is the capital of France?",
    "How do you bake a chocolate cake?",
    "What is photosynthesis?",
    "What is the largest planet?",
    "How do computers work?"
]
