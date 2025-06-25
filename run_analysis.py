#!/usr/bin/env python3
"""
Resume Intelligence System - Analysis Runner

This script provides a command-line interface to run the Resume Intelligence System pipeline.
It allows users to analyze resumes against job descriptions and generate comprehensive reports.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the pipeline
from resume_pipeline import ResumePipeline


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Resume Intelligence System - Analyze resumes against job descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
        python run_analysis.py --resume path/to/resume.pdf --job-description path/to/job.txt
        python run_analysis.py --resume path/to/resume.pdf --job-description path/to/job.txt --github-username johndoe
        """
    )
    
    # Single resume analysis arguments
    parser.add_argument("--resume", type=str, required=True, 
                       help="Path to the resume file (PDF, DOCX, or TXT)")
    parser.add_argument("--job-description", type=str, required=True,
                       help="Path to the job description file")
    parser.add_argument("--github-username", type=str,
                       help="GitHub username for additional verification")
    parser.add_argument("--linkedin-url", type=str,
                       help="LinkedIn URL for additional verification")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Directory to store analysis results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    # If no arguments are provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    return parser.parse_args()


def validate_files(args):
    """Validate that the specified files exist"""
    resume_path = Path(args.resume)
    job_desc_path = Path(args.job_description)
    
    if not resume_path.exists():
        print(f"Error: Resume file not found: {resume_path}")
        sys.exit(1)
    
    if not job_desc_path.exists():
        print(f"Error: Job description file not found: {job_desc_path}")
        sys.exit(1)


def run_analysis(args):
    """Run analysis on a single resume"""
    print(f"\nAnalyzing resume: {args.resume}")
    print(f"Against job description: {args.job_description}")
    print(f"Output directory: {args.output_dir}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Prepare contact info
    contact_info = {}
    if args.github_username:
        contact_info["github"] = args.github_username
    if args.linkedin_url:
        contact_info["linkedin"] = args.linkedin_url
    
    # Create and run pipeline
    pipeline = ResumePipeline(output_dir=str(output_dir))
    
    results = pipeline.run_pipeline(
        resume_path=args.resume,
        job_description_path=args.job_description,
        github_username=args.github_username,
        contact_info=contact_info
    )
    
    # Print summary
    print("\nAnalysis Complete")
    print("================\n")
    
    if results["status"] == "success":
        print(f"Overall Score: {results['overall_score']:.2f}%")
        print(f"Grade: {results['grade']}")
        
        print("\nComponent Scores:")
        for component, score in results["analysis_summary"].items():
            print(f"  {component.replace('_', ' ').title()}: {score:.2f}%")
        
        print("\nOutput Files:")
        for name, path in results["output_files"].items():
            print(f"  {name.replace('_', ' ').title()}: {path}")
        
        # Recommendations section removed
    
    elif results["status"] == "rejected":
        print(f"Result: REJECTED")
        print(f"Reason: {results.get('reason', 'Unknown')}")
        print(f"Skill Alignment Score: {results.get('skill_alignment_score', 0):.2f}%")
        
        if "missing_skills" in results and results["missing_skills"]:
            print("\nMissing Critical Skills:")
            for skill in results["missing_skills"]:
                print(f"  - {skill}")
        
        print("\nOutput Files:")
        for name, path in results["output_files"].items():
            print(f"  {name.replace('_', ' ').title()}: {path}")
    
    else:  # Error
        print(f"Error: {results.get('error_message', 'Unknown error')}")
    
    return results


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Validate files
    validate_files(args)
    
    # Run analysis
    run_analysis(args)


if __name__ == "__main__":
    main()