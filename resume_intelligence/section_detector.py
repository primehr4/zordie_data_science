#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Section & Structure Detector for Resume Intelligence System

This module identifies and extracts standard resume sections (education, experience, skills, etc.)
using a combination of NLP techniques and pattern matching.
"""

import json
import re
from pathlib import Path

import spacy


class SectionDetector:
    """Detector for identifying and extracting resume sections."""
    
    def __init__(self):
        """Initialize the section detector."""
        # Common section headers in resumes
        self.section_patterns = {
            'Summary': [r'(?i)\b(summary|profile|objective|about me|professional summary|career objective)\b'],
            'Education': [r'(?i)\b(education|academic|degree|university|college|educational background|academic qualifications)\b'],
            'Work Experience': [r'(?i)\b(experience|work|employment|job|career|professional|work history|professional experience|employment history)\b'],
            'Skills': [r'(?i)\b(skills|expertise|competencies|proficiencies|technical|technologies|technical skills|core competencies|key skills)\b'],
            'Projects': [r'(?i)\b(projects|portfolio|works|assignments|personal projects|project experience|key projects|technical projects|development projects)\b'],
            'Certifications': [r'(?i)\b(certifications|certificates|credentials|qualifications|professional certifications)\b'],
            'Languages': [r'(?i)\b(languages|language proficiency|spoken languages)\b'],
            'Interests': [r'(?i)\b(interests|hobbies|activities|extracurricular)\b'],
            'References': [r'(?i)\b(references|referees|professional references)\b'],
            'Publications': [r'(?i)\b(publications|papers|articles|research|research papers)\b'],
            'Awards': [r'(?i)\b(awards|honors|achievements|recognitions|accomplishments)\b'],
            'Volunteer': [r'(?i)\b(volunteer|community|service|volunteer experience|community service)\b']
        }
        
        # Load spaCy model for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            # If model is not installed, use a smaller one
            print("Warning: en_core_web_lg not found. Using en_core_web_sm instead.")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: No spaCy models found. Downloading en_core_web_sm...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
    
    def detect_sections(self, text):
        """Detect and extract sections from resume text.
        
        Args:
            text (str): The preprocessed resume text.
            
        Returns:
            dict: Dictionary mapping section names to their content.
        """
        # Find potential section headers
        potential_headers = self._find_potential_headers(text)
        
        # Print debug information about found headers
        print(f"Found {len(potential_headers)} potential section headers:")
        for section_name, start_idx, pattern in potential_headers:
            print(f"  - '{section_name}' at position {start_idx}, pattern: '{pattern}'")
        
        # If no headers found, try to extract sections based on common patterns
        if not potential_headers:
            print("No section headers found. Attempting to extract sections based on common patterns.")
            return self._extract_common_sections(text)
        
        # Extract sections based on identified headers
        sections = self._extract_sections(text, potential_headers)
        
        # Print debug information about extracted sections
        print(f"Extracted {len(sections)} sections:")
        for section_name, content in sections.items():
            content_preview = content[:50].replace('\n', ' ') + '...' if len(content) > 50 else content
            print(f"  - '{section_name}' with {len(content)} characters: '{content_preview}'")
        
        return sections
    
    def _preprocess_text(self, text):
        """Preprocess the resume text.
        
        Args:
            text (str): The raw resume text.
            
        Returns:
            str: Preprocessed text.
        """
        # Replace multiple newlines with double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Remove excessive spaces
        text = re.sub(r' {2,}', ' ', text)
        
        return text
    
    def _find_potential_headers(self, text):
        """Find potential section headers in the text.
        
        Args:
            text (str): The preprocessed resume text.
            
        Returns:
            list: List of tuples (section_name, start_index, pattern_matched).
        """
        potential_headers = []
        
        # Split text into lines
        lines = text.split('\n')
        
        # Track current position in text
        current_pos = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                current_pos += 1  # Account for newline
                continue
            
            # Check if line matches any section pattern
            for section_name, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Enhanced header detection criteria
                        is_header = False
                        
                        # Standard header formats
                        if len(line) < 50 and (line.endswith(':') or line.isupper() or line.istitle()):
                            is_header = True
                        
                        # Check for standalone words that are likely headers
                        elif len(line.split()) <= 3 and len(line) < 30:
                            # Check if the line is followed by a blank line or bullet points
                            if current_pos + len(line) + 1 < len(text):
                                next_line = text[current_pos + len(line) + 1:].split('\n', 1)[0].strip()
                                if not next_line or next_line.startswith(('•', '-', '*', '\t', '    ')):
                                    is_header = True
                        
                        # Check for centered text (potential header)
                        elif line.strip() == line and len(line) < 30:
                            surrounding_lines = [l.strip() for l in lines[max(0, lines.index(line)-2):min(len(lines), lines.index(line)+3)]]
                            if all(len(l) < len(line) or not l for l in surrounding_lines if l != line):
                                is_header = True
                                
                        # Check for lines with formatting that suggests a header
                        # (e.g., all caps, title case, or followed by a line of dashes/underscores)
                        elif line.isupper() or (line.istitle() and len(line.split()) <= 4):
                            is_header = True
                        elif current_pos + len(line) + 1 < len(text):
                            next_line = text[current_pos + len(line) + 1:].split('\n', 1)[0].strip()
                            if next_line and all(c in '-_=' for c in next_line):
                                is_header = True
                        
                        if is_header:
                            # Use the full section name from the dictionary, not a truncated version
                            potential_headers.append((section_name, current_pos, line))
                            break
            
            current_pos += len(line) + 1  # +1 for newline
        
        # Sort headers by position in text
        potential_headers.sort(key=lambda x: x[1])
        
        return potential_headers
    
    def _extract_sections(self, text, headers):
        """Extract content between identified headers.
        
        Args:
            text (str): The preprocessed resume text.
            headers (list): List of tuples (section_name, start_index, pattern_matched).
            
        Returns:
            dict: Dictionary of extracted sections.
        """
        sections = {}
        
        # Extract content between headers
        for i, (section_name, start_idx, pattern) in enumerate(headers):
            # Find the end of the current section (start of next section or end of text)
            end_idx = headers[i+1][1] if i < len(headers) - 1 else len(text)
            
            # Extract content
            content = text[start_idx:end_idx].strip()
            
            # Remove the header line from the content
            content_lines = content.split('\n')
            if content_lines and content_lines[0].strip() == pattern.strip():
                content = '\n'.join(content_lines[1:]).strip()
            
            # Print debug information
            print(f"Extracted section: '{section_name}' with {len(content)} characters")
            print(f"Content preview: '{content[:50]}...'")
            
            sections[section_name] = content
        
        return sections
    
    def _post_process_sections(self, sections, text):
        """Apply post-processing to improve section detection.
        
        Args:
            sections (dict): Dictionary of detected sections.
            text (str): The full resume text.
            
        Returns:
            dict: Improved sections dictionary.
        """
        # If no sections were detected, try to extract them using NLP
        if not sections:
            sections = self._extract_sections_with_nlp(text)
        
        # Ensure we have a skills section
        if 'Skills' not in sections:
            skills_text = self._extract_skills_section(text)
            if skills_text:
                sections['Skills'] = skills_text
        
        # Extract education if not already present
        if 'Education' not in sections:
            education_text = self._extract_education_section(text)
            if education_text:
                sections['Education'] = education_text
        
        # Extract projects if not already present
        if 'Projects' not in sections:
            projects_text = self._extract_projects_section(text)
            if projects_text:
                sections['Projects'] = projects_text
        
        return sections
    
    def _fallback_section_detection(self, text):
        """Fallback method for section detection when header-based approach fails.
        
        Args:
            text (str): The original resume text.
            
        Returns:
            dict: Dictionary of detected sections.
        """
        sections = {}
        
        # Try to identify sections based on formatting patterns first
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        # First pass: look for clear section headers
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Enhanced header detection
            is_header = False
            
            # Standard header formats (all caps, title case, ends with colon)
            if (line.isupper() or line.istitle() or line.endswith(':')) and len(line) < 30:
                is_header = True
            
            # Check for lines that are followed by bullet points or indented text
            elif len(line) < 30 and i < len(lines) - 1:
                next_line = lines[i+1].strip() if i+1 < len(lines) else ''
                if next_line.startswith(('•', '-', '*', '\t')) or (next_line and next_line[0].isspace()):
                    is_header = True
            
            # Check for centered text (potential header)
            elif len(line) < 30 and i > 0 and i < len(lines) - 1:
                prev_line = lines[i-1].strip()
                next_line = lines[i+1].strip()
                if (not prev_line or len(prev_line) < len(line)) and (not next_line or len(next_line) < len(line)):
                    is_header = True
            
            if is_header:
                # Save previous section if exists
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section with improved section type guessing
                current_section = self._guess_section_type(line)
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        # If formatting-based approach didn't work well, use NLP approach
        if not sections or len(sections) < 2:
            # Use NLP to identify potential sections
            doc = self.nlp(text)
            
            # Look for skill-related keywords
            skill_keywords = ['skills', 'technologies', 'tools', 'languages', 'frameworks']
            skill_sentences = [sent.text for sent in doc.sents 
                              if any(keyword in sent.text.lower() for keyword in skill_keywords)]
            
            if skill_sentences:
                # Extract a few sentences around skill mentions as the Skills section
                skills_text = ' '.join(skill_sentences)
                sections['Skills'] = skills_text
            
            # Look for education-related keywords
            edu_keywords = ['degree', 'university', 'college', 'bachelor', 'master', 'phd', 'diploma']
            edu_sentences = [sent.text for sent in doc.sents 
                            if any(keyword in sent.text.lower() for keyword in edu_keywords)]
            
            if edu_sentences:
                sections['Education'] = ' '.join(edu_sentences)
            
            # Look for experience-related keywords
            exp_keywords = ['experience', 'work', 'job', 'position', 'role', 'company', 'employer']
            exp_sentences = [sent.text for sent in doc.sents 
                            if any(keyword in sent.text.lower() for keyword in exp_keywords)]
            
            if exp_sentences:
                sections['Work Experience'] = ' '.join(exp_sentences)
            
            # Look for project-related keywords
            proj_keywords = ['project', 'developed', 'created', 'built', 'implemented', 'designed', 'github']
            proj_sentences = [sent.text for sent in doc.sents 
                            if any(keyword in sent.text.lower() for keyword in proj_keywords)]
            
            if proj_sentences:
                sections['Projects'] = ' '.join(proj_sentences)
        
        return sections
    
    def _extract_skills_section(self, text):
        """Extract skills section using pattern matching.
        
        Args:
            text (str): The resume text.
            
        Returns:
            str: Extracted skills section or empty string if not found.
        """
        # Look for common skills section patterns
        skills_patterns = [
            r'(?i)skills[:\s]*(.*?)(?:\n\n|\Z)',
            r'(?i)technical skills[:\s]*(.*?)(?:\n\n|\Z)',
            r'(?i)technologies[:\s]*(.*?)(?:\n\n|\Z)'
        ]
        
        for pattern in skills_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def save_sections(self, sections, output_path):
        """Save extracted sections to a JSON file.
        
        Args:
            sections (dict): Dictionary of extracted sections.
            output_path (str or Path): Path to save the JSON file.
        """
        # Create a copy of the sections dictionary to avoid modifying the original
        sections_to_save = {}
        
        # Check for truncated section names and fix them
        for section_name, content in sections.items():
            # Print debug information
            print(f"Checking section: '{section_name}' with {len(content)} characters")
            print(f"Content preview: '{content[:50]}...'")
            
            # Check if this might be a truncated section name
            fixed_name = section_name
            
            # More aggressive check for truncated section names
            for full_name in self.section_patterns.keys():
                # Check if section_name is a prefix of full_name (case-insensitive)
                if full_name.lower().startswith(section_name.lower()) and section_name.lower() != full_name.lower():
                    print(f"Found potential match for truncated section name: '{section_name}' -> '{full_name}'")
                    fixed_name = full_name
                    break
                # Check if section_name is a short form or abbreviation (case-insensitive)
                elif len(section_name) <= 6 and section_name.lower() in full_name.lower():
                    print(f"Found potential match for abbreviated section name: '{section_name}' -> '{full_name}'")
                    fixed_name = full_name
                    break
            
            # Use the fixed or original section name
            if fixed_name != section_name:
                print(f"Fixing section name: '{section_name}' -> '{fixed_name}'")
            sections_to_save[fixed_name] = content
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sections_to_save, f, indent=2)
            
        # Verify the saved sections
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_sections = json.load(f)
            for section_name, content in saved_sections.items():
                print(f"Verified saved section: '{section_name}' with {len(content)} characters")
                print(f"Content preview: '{content[:50]}...'")
                if len(content) < 20 and section_name != "Skills":
                    print(f"WARNING: Section '{section_name}' content appears truncated: '{content}'")
                    # Try to recover the original content
                    original_name = section_name
                    for name in sections.keys():
                        if name.lower().startswith(section_name.lower()) or section_name.lower().startswith(name.lower()):
                            original_name = name
                            break
                    
                    if original_name in sections:
                        print(f"Original content length: {len(sections[original_name])}")
                        print(f"Original content preview: '{sections[original_name][:50]}...'")
                        # Attempt to fix the truncated content
                        with open(output_path, 'w', encoding='utf-8') as fix_f:
                            sections_to_save[section_name] = sections[original_name]
                            json.dump(sections_to_save, fix_f, indent=2)
                        print(f"Attempted to fix truncated content for section '{section_name}'")
    
    def load_sections(self, input_path):
        """Load sections from a JSON file.
        
        Args:
            input_path (str or Path): Path to the JSON file.
            
        Returns:
            dict: Dictionary of sections.
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def analyze_structure(self, sections):
        """Analyze the structure of the resume sections.
        
        This method evaluates the completeness and quality of each resume section,
        flags missing or sparse sections, and provides an overall structure score.
        
        Args:
            sections (dict): Dictionary mapping section names to their content.
            
        Returns:
            dict: Analysis results including structure score, missing sections, and sparse sections.
        """
        # Define expected sections and their importance (weight)
        expected_sections = {
            'Summary': 0.10,
            'Education': 0.15,
            'Work Experience': 0.25,
            'Skills': 0.20,
            'Projects': 0.15,
            'Certifications': 0.05,
            'Languages': 0.05,
            'Interests': 0.05
        }
        
        # Define minimum content length for each section (in characters)
        min_section_length = {
            'Summary': 100,
            'Education': 100,
            'Work Experience': 200,
            'Skills': 100,
            'Projects': 150,
            'Certifications': 50,
            'Languages': 30,
            'Interests': 30
        }
        
        # Initialize results
        missing_sections = []
        sparse_sections = []
        section_scores = {}
        
        # Calculate base score for each section
        total_weight = 0
        total_score = 0
        
        # Evaluate each expected section
        for section_name, weight in expected_sections.items():
            if section_name not in sections:
                missing_sections.append(section_name)
                section_scores[section_name] = 0
            else:
                content = sections[section_name]
                content_length = len(content)
                
                # Check if section is sparse
                if content_length < min_section_length.get(section_name, 50):
                    sparse_sections.append(section_name)
                    # Partial score based on content length
                    section_score = (content_length / min_section_length.get(section_name, 50)) * 100
                else:
                    # Full score for adequate content
                    section_score = 100
                
                # Apply weight to section score
                section_scores[section_name] = section_score
                total_score += section_score * weight
                
            total_weight += weight
        
        # Normalize total score if we have weights
        if total_weight > 0:
            structure_score = total_score / total_weight
        else:
            structure_score = 0
        
        # Check for unexpected but valuable sections
        for section_name in sections:
            if section_name not in expected_sections and len(sections[section_name]) > 100:
                # Add a small bonus for additional valuable sections
                structure_score += 2
        
        # Cap the score at 100
        structure_score = min(structure_score, 100)
        
        return {
            'structure_score': structure_score,
            'missing_sections': missing_sections,
            'sparse_sections': sparse_sections,
            'section_scores': section_scores
        }