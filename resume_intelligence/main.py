import os
import sys
import logging
import argparse
from pathlib import Path
import uvicorn
import threading
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

# Import system components
from resume_intelligence.api_layer import app as api_app
from resume_intelligence.workflow_orchestration import WorkflowOrchestrator


def start_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server"""
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run("resume_intelligence.api_layer:app", host=host, port=port, reload=reload)


def start_workflow_orchestrator(output_dir: str = "./workflow_output"):
    """Start the workflow orchestrator"""
    logger.info("Starting workflow orchestrator")
    orchestrator = WorkflowOrchestrator(output_dir=output_dir)
    orchestrator.start()
    return orchestrator


def main():
    """Main entry point for the Resume Intelligence System"""
    parser = argparse.ArgumentParser(description="Resume Intelligence System")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Workflow orchestrator command
    workflow_parser = subparsers.add_parser("workflow", help="Start the workflow orchestrator")
    workflow_parser.add_argument("--output-dir", type=str, default="./workflow_output", 
                               help="Output directory for workflow results")
    
    # Full system command
    full_parser = subparsers.add_parser("full", help="Start the full system (API + workflow)")
    full_parser.add_argument("--api-host", type=str, default="0.0.0.0", help="API host to bind to")
    full_parser.add_argument("--api-port", type=int, default=8000, help="API port to bind to")
    full_parser.add_argument("--workflow-output-dir", type=str, default="./workflow_output", 
                           help="Output directory for workflow results")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "api":
        start_api_server(host=args.host, port=args.port, reload=args.reload)
    
    elif args.command == "workflow":
        orchestrator = start_workflow_orchestrator(output_dir=args.output_dir)
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping workflow orchestrator")
            orchestrator.stop()
    
    elif args.command == "full":
        # Start workflow orchestrator in a separate thread
        orchestrator = start_workflow_orchestrator(output_dir=args.workflow_output_dir)
        
        # Start API server in main thread
        try:
            start_api_server(host=args.api_host, port=args.api_port)
        except KeyboardInterrupt:
            logger.info("Stopping system")
            orchestrator.stop()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()