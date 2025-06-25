import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

class QualityScoreEngine:
    """
    Final Quality Score & Issue Report: Fuse all sub-scores into a unified 
    résumé score (0–100) and emit a structured JSON report.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for quality scoring"""
        return {
            "weights": {
                "skill_alignment": 0.30,  # 30% - Recalculated Skill Alignment
                "project_validation": 0.30,  # 30% - Project Validation
                "formatting": 0.10,  # 10% - Resume Formatting
                "trustworthiness": 0.10,  # 10% - Content Trustworthiness
                "credibility": 0.10,  # 10% - Credential Verification
                "online_presence": 0.10  # 10% - Online Presence
            },
            "grade_thresholds": {
                "A+": 95,
                "A": 90,
                "A-": 85,
                "B+": 80,
                "B": 75,
                "B-": 70,
                "C+": 65,
                "C": 60,
                "C-": 55,
                "D": 50,
                "F": 0
            },
            "critical_issues_threshold": 3,
            "warning_issues_threshold": 5
        }
    
    def calculate_final_quality_score(self, 
                                    skill_alignment_results: Dict[str, Any],
                                    project_validation_results: Dict[str, Any],
                                    formatting_results: Dict[str, Any],
                                    trustworthiness_results: Dict[str, Any],
                                    credibility_results: Dict[str, Any],
                                    resume_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate final quality score and generate comprehensive report
        
        Args:
            skill_alignment_results: Results from skill matcher
            project_validation_results: Results from project validator
            formatting_results: Results from formatting scorer
            trustworthiness_results: Results from trustworthiness detector
            credibility_results: Results from credibility engine
            resume_metadata: Additional metadata about the resume
            
        Returns:
            Comprehensive quality report with final score
        """
        try:
            # Extract individual scores
            scores = self._extract_component_scores(
                skill_alignment_results,
                project_validation_results,
                formatting_results,
                trustworthiness_results,
                credibility_results
            )
            
            # Add online presence score if available in metadata
            if resume_metadata and isinstance(resume_metadata, dict):
                scores["online_presence"] = resume_metadata.get("online_presence_score", 0)
            
            # Calculate weighted final score
            final_score = self._calculate_weighted_score(scores)
            
            # Determine grade
            grade = self._determine_grade(final_score)
            
            # Collect all issues and flags
            issues = self._collect_issues(
                skill_alignment_results,
                project_validation_results,
                formatting_results,
                trustworthiness_results,
                credibility_results
            )
            
            # Generate recommendations
            recommendations = self._generate_comprehensive_recommendations(
                scores, issues, skill_alignment_results, project_validation_results
            )
            
            # Create detailed report
            quality_report = {
                "overall_score": round(final_score, 2),
                "grade": grade,
                "max_score": 100,
                "analysis_timestamp": datetime.now().isoformat(),
                "component_scores": scores,
                "score_breakdown": self._create_score_breakdown(scores),
                "issues_summary": self._summarize_issues(issues),
                "detailed_issues": issues,
                "recommendations": recommendations,
                "strengths": self._identify_strengths(scores, skill_alignment_results, project_validation_results),
                "improvement_areas": self._identify_improvement_areas(scores, issues),
                "metadata": resume_metadata or {}
            }
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return self._get_default_quality_report()
    
    def _extract_component_scores(self, skill_alignment: Dict, project_validation: Dict,
                                formatting: Dict, trustworthiness: Dict, 
                                credibility: Dict) -> Dict[str, float]:
        """Extract normalized scores from each component"""
        scores = {}
        
        # Skill alignment score (already 0-100)
        scores["skill_alignment"] = skill_alignment.get("overall_alignment_score", 0)
        
        # Project validation score (convert to 0-100 if needed)
        project_score = 0
        # Handle different types of project_validation results
        if isinstance(project_validation, dict):
            # Try to get overall_score directly
            if "overall_score" in project_validation:
                project_score = project_validation.get("overall_score", 0)
                # If overall_score is a dict, extract total
                if isinstance(project_score, dict):
                    project_score = project_score.get("total", 0)
            # If no overall_score, calculate from project_scores
            elif "project_scores" in project_validation and project_validation["project_scores"]:
                project_scores = project_validation.get("project_scores", {})
                if project_scores:
                    # Calculate average project score and scale to 0-100
                    try:
                        project_score = sum(project_scores.values()) / len(project_scores) * 100
                    except (TypeError, ValueError):
                        # Handle case where project_scores values are not numeric
                        project_score = 0
        
        # Ensure project_score is a number
        try:
            project_score = float(project_score)
        except (ValueError, TypeError):
            project_score = 0
            
        scores["project_validation"] = min(100, max(0, project_score))
        
        # Formatting score (convert to 0-100)
        formatting_score = formatting.get("total_score", 0)
        formatting_max = formatting.get("max_score", 20)
        scores["formatting"] = (formatting_score / max(formatting_max, 1)) * 100
        
        # Trustworthiness score (already 0-100)
        scores["trustworthiness"] = trustworthiness.get("trust_score", 50)
        
        # Credibility score (already 0-100)
        scores["credibility"] = credibility.get("credibility_score", 50)
        
        return scores
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted final score"""
        weighted_score = 0.0
        total_weight = 0.0
        
        for component, score in scores.items():
            weight = self.config["weights"].get(component, 0)
            weighted_score += score * weight
            total_weight += weight
        
        # Normalize if weights don't sum to 1
        if total_weight > 0:
            weighted_score = weighted_score / total_weight
        
        return min(100, max(0, weighted_score))
    
    def _determine_grade(self, score: float) -> str:
        """Determine letter grade based on score"""
        for grade, threshold in self.config["grade_thresholds"].items():
            if score >= threshold:
                return grade
        return "F"
    
    def _collect_issues(self, skill_alignment: Dict, project_validation: Dict,
                       formatting: Dict, trustworthiness: Dict, 
                       credibility: Dict) -> List[Dict[str, Any]]:
        """Collect all issues and flags from components"""
        all_issues = []
        
        # Skill alignment issues
        missing_skills = skill_alignment.get("missing_skills", [])
        for skill in missing_skills:
            all_issues.append({
                "type": "missing_skill",
                "severity": "medium",
                "component": "skill_alignment",
                "message": f"Missing skill: {skill}",
                "recommendation": f"Consider adding experience or projects demonstrating {skill}"
            })
        
        # Project validation issues
        flagged_projects = project_validation.get("flagged_projects", [])
        for project in flagged_projects:
            # Check if project is a string or a dictionary
            if isinstance(project, str):
                # Handle string format
                all_issues.append({
                    "type": "project_issue",
                    "severity": "medium",
                    "component": "project_validation",
                    "message": f"Project issue: {project}",
                    "recommendation": "Improve project description with specific technologies and quantifiable results"
                })
            else:
                # Handle dictionary format
                all_issues.append({
                    "type": "project_issue",
                    "severity": "medium",
                    "component": "project_validation",
                    "message": f"Project '{project.get('title', 'Unknown')}' flagged for review",
                    "details": project.get("issues", []),
                    "recommendation": "Improve project description with specific technologies and quantifiable results"
                })
        
        # Formatting issues
        formatting_recommendations = formatting.get("recommendations", [])
        for rec in formatting_recommendations:
            if "excellent" not in rec.lower():
                all_issues.append({
                    "type": "formatting_issue",
                    "severity": "low",
                    "component": "formatting",
                    "message": rec,
                    "recommendation": rec
                })
        
        # Trustworthiness flags
        trust_flags = trustworthiness.get("flags", [])
        for flag in trust_flags:
            all_issues.append({
                "type": flag.get("type", "trust_issue"),
                "severity": flag.get("severity", "medium"),
                "component": "trustworthiness",
                "message": flag.get("message", "Trustworthiness concern detected"),
                "recommendation": "Review and improve content authenticity"
            })
        
        # Credibility issues
        credibility_recommendations = credibility.get("recommendations", [])
        for rec in credibility_recommendations:
            if "strong credibility" not in rec.lower():
                all_issues.append({
                    "type": "credibility_issue",
                    "severity": "low",
                    "component": "credibility",
                    "message": rec,
                    "recommendation": rec
                })
        
        return all_issues
    
    def _summarize_issues(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize issues by type and severity"""
        summary = {
            "total_issues": len(issues),
            "by_severity": {"high": 0, "medium": 0, "low": 0},
            "by_component": {},
            "critical_issues": [],
            "status": "good"
        }
        
        for issue in issues:
            severity = issue.get("severity", "medium")
            component = issue.get("component", "unknown")
            
            # Count by severity
            summary["by_severity"][severity] += 1
            
            # Count by component
            if component not in summary["by_component"]:
                summary["by_component"][component] = 0
            summary["by_component"][component] += 1
            
            # Collect critical issues
            if severity == "high":
                summary["critical_issues"].append(issue["message"])
        
        # Determine overall status
        high_issues = summary["by_severity"]["high"]
        total_issues = summary["total_issues"]
        
        if high_issues >= self.config["critical_issues_threshold"]:
            summary["status"] = "critical"
        elif total_issues >= self.config["warning_issues_threshold"]:
            summary["status"] = "needs_improvement"
        elif total_issues > 0:
            summary["status"] = "minor_issues"
        else:
            summary["status"] = "excellent"
        
        return summary
    
    def _create_score_breakdown(self, scores: Dict[str, float]) -> Dict[str, Any]:
        """Create detailed score breakdown with weights"""
        breakdown = {}
        
        # Define component display names for better readability
        component_display_names = {
            "skill_alignment": "Skill Alignment",
            "project_validation": "Project Validation",
            "formatting": "Resume Formatting",
            "trustworthiness": "Content Trustworthiness",
            "credibility": "Credential Verification",
            "online_presence": "Online Presence"
        }
        
        for component, score in scores.items():
            weight = self.config["weights"].get(component, 0)
            weighted_contribution = score * weight
            
            breakdown[component] = {
                "raw_score": round(score, 2),
                "weight": weight,
                "weighted_contribution": round(weighted_contribution, 2),
                "percentage_of_total": round((weight * 100), 1),
                "display_name": component_display_names.get(component, component.replace("_", " ").title())
            }
        
        return breakdown
    
    def _generate_comprehensive_recommendations(self, scores: Dict[str, float], 
                                              issues: List[Dict[str, Any]],
                                              skill_alignment: Dict[str, Any],
                                              project_validation: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate comprehensive recommendations"""
        recommendations = {
            "immediate_actions": [],
            "skill_development": [],
            "content_improvement": [],
            "presentation_enhancement": [],
            "credibility_building": []
        }
        
        # Safe access to nested dictionaries
        def safe_get(d, *keys, default=None):
            """Safely get a value from nested dictionaries"""
            if not isinstance(d, dict):
                return default
            
            current = d
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return default
                current = current[key]
            
            return current
        
        # Immediate actions based on critical issues
        high_severity_issues = [issue for issue in issues if issue.get("severity") == "high"]
        for issue in high_severity_issues:
            recommendations["immediate_actions"].append(issue.get("recommendation", issue.get("message")))
        
        # Skill development recommendations
        missing_skills = skill_alignment.get("missing_skills", [])
        if missing_skills:
            recommendations["skill_development"].extend([
                f"Develop proficiency in {skill}" for skill in missing_skills[:5]  # Top 5
            ])
        
        # Content improvement based on project validation
        flagged_projects = project_validation.get("flagged_projects", [])
        if flagged_projects:
            recommendations["content_improvement"].extend([
                "Add quantifiable results to project descriptions",
                "Include specific technologies and methodologies used",
                "Highlight problem-solving approaches and outcomes"
            ])
        
        # Presentation enhancement based on formatting score
        if scores.get("formatting", 100) < 70:
            recommendations["presentation_enhancement"].extend([
                "Improve resume formatting and visual consistency",
                "Use professional fonts and appropriate spacing",
                "Ensure consistent bullet point usage"
            ])
        
        # Credibility building based on scores
        if scores.get("credibility", 100) < 60:
            recommendations["credibility_building"].extend([
                "Add professional certifications with verification details",
                "Include links to professional profiles (LinkedIn, GitHub)",
                "Ensure consistency across all professional platforms"
            ])
        
        # Remove duplicates and empty categories
        for category in recommendations:
            recommendations[category] = list(set(recommendations[category]))
            if not recommendations[category]:
                recommendations[category] = ["No specific recommendations - maintain current standards"]
        
        return recommendations
    
    def _identify_weaknesses(self, scores: Dict[str, float], 
                           issues: List[Dict[str, Any]],
                           skill_alignment: Dict[str, Any],
                           project_validation: Dict[str, Any]) -> List[str]:
        """Identify resume weaknesses"""
        weaknesses = []
        
        # Safe access to nested dictionaries
        def safe_get(d, *keys, default=None):
            """Safely get a value from nested dictionaries"""
            if not isinstance(d, dict):
                return default
            
            current = d
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return default
                current = current[key]
            
            return current
        
        # Score-based weaknesses
        for component, score in scores.items():
            if score < 50:
                component_name = component.replace("_", " ").title()
                weaknesses.append(f"Poor {component_name} (Score: {score:.1f}/100)")
            elif score < 65:
                component_name = component.replace("_", " ").title()
                weaknesses.append(f"Below average {component_name} (Score: {score:.1f}/100)")
        
        # Skill-based weaknesses
        missing_skills = safe_get(skill_alignment, "missing_skills", default=[])
        if missing_skills and len(missing_skills) >= 3:
            weaknesses.append(f"Missing {len(missing_skills)} important job requirements")
        
        # Project-based weaknesses
        project_issues = safe_get(project_validation, "issues", default=[])
        if project_issues:
            weaknesses.append(f"Has {len(project_issues)} issues with project documentation")
        
        return weaknesses
    
    def _identify_strengths(self, scores: Dict[str, float], 
                          skill_alignment: Dict[str, Any],
                          project_validation: Dict[str, Any]) -> List[str]:
        """Identify resume strengths"""
        strengths = []
        
        # Safe access to nested dictionaries
        def safe_get(d, *keys, default=None):
            """Safely get a value from nested dictionaries"""
            if not isinstance(d, dict):
                return default
            
            current = d
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return default
                current = current[key]
            
            return current
        
        # Score-based strengths
        for component, score in scores.items():
            if score >= 85:
                component_name = component.replace("_", " ").title()
                strengths.append(f"Excellent {component_name} (Score: {score:.1f}/100)")
            elif score >= 75:
                component_name = component.replace("_", " ").title()
                strengths.append(f"Strong {component_name} (Score: {score:.1f}/100)")
        
        # Skill-based strengths
        matched_skills = safe_get(skill_alignment, "matched_skills", default=[])
        if matched_skills and len(matched_skills) >= 3:
            strengths.append(f"Strong alignment with {len(matched_skills)} job requirements")
        
        # Project-based strengths
        top_projects = safe_get(project_validation, "top_projects", default=[])
        if top_projects:
            strengths.append(f"Has {len(top_projects)} well-documented projects")
        
        return strengths
    
    def _identify_improvement_areas(self, scores: Dict[str, float], 
                                   issues: List[Dict[str, Any]]) -> List[str]:
        """Identify areas for improvement"""
        improvement_areas = []
        
        # Safe access to nested dictionaries
        def safe_get(d, *keys, default=None):
            """Safely get a value from nested dictionaries"""
            if not isinstance(d, dict):
                return default
            
            current = d
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    return default
                current = current[key]
            
            return current
        
        # Score-based improvement areas
        for component, score in scores.items():
            if 65 <= score < 75:
                component_name = component.replace("_", " ").title()
                improvement_areas.append(f"{component_name} needs enhancement (Score: {score:.1f}/100)")
        
        # Issue-based improvement areas (medium severity)
        medium_severity_issues = [issue for issue in issues if issue.get("severity") == "medium"]
        if medium_severity_issues:
            for issue in medium_severity_issues[:3]:  # Limit to top 3
                improvement_areas.append(issue.get("description", "Unknown issue"))
        
        return improvement_areas
    
    def save_quality_report(self, quality_report: Dict[str, Any], 
                          output_path: str = "quality_report.json") -> bool:
        """Save quality report to JSON file"""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            
            # Convert numpy types to Python native types to ensure JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                                     np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return convert_numpy_types(obj.tolist())
                else:
                    return obj
            
            # Convert numpy types in the quality report
            converted_report = convert_numpy_types(quality_report)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Quality report saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving quality report: {e}")
            return False
    
    def generate_summary_report(self, quality_report: Dict[str, Any]) -> str:
        """Generate human-readable summary report"""
        try:
            summary = []
            summary.append("=" * 60)
            summary.append("RESUME QUALITY ANALYSIS REPORT")
            summary.append("=" * 60)
            summary.append("")
            
            # Overall score
            overall_score = quality_report.get("overall_score", 0)
            grade = quality_report.get("grade", "F")
            summary.append(f"Overall Score: {overall_score:.1f}/100 (Grade: {grade})")
            summary.append("")
            
            # Component scores
            summary.append("Component Breakdown:")
            summary.append("-" * 30)
            score_breakdown = quality_report.get("score_breakdown", {})
            for component, details in score_breakdown.items():
                component_name = details.get("display_name", component.replace("_", " ").title())
                raw_score = details.get("raw_score", 0)
                weight = details.get("percentage_of_total", 0)
                summary.append(f"{component_name:20} {raw_score:6.1f}/100 ({weight:4.1f}% weight)")
            summary.append("")
            
            # Issues summary
            issues_summary = quality_report.get("issues_summary", {})
            total_issues = issues_summary.get("total_issues", 0)
            status = issues_summary.get("status", "unknown")
            summary.append(f"Issues Summary: {total_issues} total issues (Status: {status.upper()})")
            
            by_severity = issues_summary.get("by_severity", {})
            summary.append(f"  High: {by_severity.get('high', 0)}, Medium: {by_severity.get('medium', 0)}, Low: {by_severity.get('low', 0)}")
            summary.append("")
            
            # Strengths
            strengths = quality_report.get("strengths", [])
            if strengths:
                summary.append("Key Strengths:")
                summary.append("-" * 15)
                for strength in strengths:
                    summary.append(f"• {strength}")
                summary.append("")
            
            # Improvement areas
            improvement_areas = quality_report.get("improvement_areas", [])
            if improvement_areas:
                summary.append("Areas for Improvement:")
                summary.append("-" * 25)
                for area in improvement_areas:
                    summary.append(f"• {area}")
                summary.append("")
            
            # Top recommendations
            recommendations = quality_report.get("recommendations", {})
            immediate_actions = recommendations.get("immediate_actions", [])
            if immediate_actions:
                summary.append("Immediate Action Items:")
                summary.append("-" * 25)
                for action in immediate_actions[:5]:  # Top 5
                    summary.append(f"• {action}")
                summary.append("")
            
            summary.append("=" * 60)
            summary.append(f"Report generated: {quality_report.get('analysis_timestamp', 'Unknown')}")
            summary.append("=" * 60)
            
            return "\n".join(summary)
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return "Error generating summary report"
    
    def _get_default_quality_report(self) -> Dict[str, Any]:
        """Return default quality report when analysis fails"""
        return {
            "overall_score": 0.0,
            "grade": "F",
            "max_score": 100,
            "analysis_timestamp": datetime.now().isoformat(),
            "component_scores": {},
            "score_breakdown": {},
            "issues_summary": {"total_issues": 0, "status": "error"},
            "detailed_issues": [],
            "recommendations": {"immediate_actions": ["Could not analyze resume - please check file format and content"]},
            "strengths": [],
            "improvement_areas": ["Analysis failed - unable to determine improvement areas"],
            "metadata": {}
        }

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate overall score based on component scores"""
        # Define weights for each component
        weights = {
            "content": 0.25,
            "formatting": 0.15,
            "ats_compatibility": 0.20,
            "skill_alignment": 0.25,
            "project_validation": 0.15
        }
        
        # Calculate weighted score
        weighted_sum = 0
        total_weight = 0
        
        for component, weight in weights.items():
            if component in scores and isinstance(scores[component], (int, float)):
                weighted_sum += scores[component] * weight
                total_weight += weight
        
        # If no valid scores were found, return 0
        if total_weight == 0:
            return 0
        
        # Calculate final score (normalized by total weight)
        overall_score = weighted_sum / total_weight
        
        # Round to 1 decimal place
        return round(overall_score, 1)