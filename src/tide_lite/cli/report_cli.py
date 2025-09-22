#!/usr/bin/env python3
"""
Report Generator CLI for TIDE-Lite
Fills the report template with actual results from experiments.
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate markdown reports from experimental results."""
    
    def __init__(self, template_path: Path, output_path: Path):
        """
        Initialize the report generator.
        
        Args:
            template_path: Path to the report template
            output_path: Path for the generated report
        """
        self.template_path = template_path
        self.output_path = output_path
        self.results_dir = Path("results")
        self.plots_dir = Path("results/plots")
        
    def load_summary(self) -> Optional[Dict[str, Any]]:
        """Load the summary JSON file if it exists."""
        summary_path = self.results_dir / "summary.json"
        
        if not summary_path.exists():
            logger.warning(f"Summary file not found: {summary_path}")
            return None
            
        try:
            with open(summary_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error loading summary JSON: {e}")
            return None
            
    def load_template(self) -> str:
        """Load the report template."""
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")
            
        with open(self.template_path, 'r') as f:
            return f.read()
            
    def fill_results_table(self, template: str, summary: Optional[Dict]) -> str:
        """Fill in the results table with actual data."""
        if not summary:
            return template.replace("[Placeholder]", "TODO: Run experiments")
            
        # Extract results if available
        results = summary.get("results", {})
        
        # Build table rows
        rows = []
        for method, metrics in results.items():
            row = f"| {method} "
            row += f"| {metrics.get('quora_spearman', '-'):.3f} " if 'quora_spearman' in metrics else "| - "
            row += f"| {metrics.get('stsb_pearson', '-'):.3f} " if 'stsb_pearson' in metrics else "| - "
            row += f"| {metrics.get('news_mrr10', '-'):.3f} " if 'news_mrr10' in metrics else "| - "
            row += f"| {metrics.get('inference_ms', '-'):.1f} " if 'inference_ms' in metrics else "| - "
            row += f"| {metrics.get('memory_gb', '-'):.2f} |" if 'memory_gb' in metrics else "| - |"
            rows.append(row)
            
        if rows:
            # Find the placeholder table and replace
            table_placeholder = "| [Placeholder] | - | - | - | - | - |"
            table_content = "\n".join(rows)
            template = template.replace(table_placeholder, table_content)
            
        return template
        
    def update_plots(self, template: str) -> str:
        """Update plot references with actual file paths."""
        plots_mapping = {
            "![Placeholder for correlation_drift.png]": "",
            "![Placeholder for ablation_results.png]": "",
            "![Placeholder for memory_tradeoff.png]": ""
        }
        
        # Check for actual plots
        if self.plots_dir.exists():
            for plot_file in self.plots_dir.glob("*.png"):
                if "correlation" in plot_file.name.lower():
                    plots_mapping["![Placeholder for correlation_drift.png]"] = \
                        f"![Correlation vs Time Drift]({plot_file.relative_to(Path.cwd())})"
                elif "ablation" in plot_file.name.lower():
                    plots_mapping["![Placeholder for ablation_results.png]"] = \
                        f"![Ablation Study Results]({plot_file.relative_to(Path.cwd())})"
                elif "memory" in plot_file.name.lower():
                    plots_mapping["![Placeholder for memory_tradeoff.png]"] = \
                        f"![Memory-Performance Tradeoff]({plot_file.relative_to(Path.cwd())})"
                        
        # Replace placeholders
        for placeholder, replacement in plots_mapping.items():
            if replacement:
                template = template.replace(placeholder, replacement)
            else:
                template = template.replace(placeholder, 
                    f"{placeholder}\n**TODO**: Generate plot")
                
        return template
        
    def update_metadata(self, template: str) -> str:
        """Update report metadata (date, version, etc.)."""
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        template = template.replace("[DATE]", current_date)
        
        # Update version based on git or incrementally
        # For now, keep it simple
        template = template.replace("Version: 1.0", "Version: 1.0-auto")
        
        return template
        
    def add_todos(self, template: str, summary: Optional[Dict]) -> str:
        """Add TODO markers for missing data."""
        if not summary:
            # Add prominent TODO at the top
            todo_section = """
> **⚠️ TODO**: No experimental results found!
> 
> Please run the following commands:
> ```bash
> python -m tide_lite.cli.train_cli
> python -m tide_lite.cli.eval_stsb_cli
> python -m tide_lite.cli.eval_quora_cli
> python -m tide_lite.cli.eval_temporal_cli
> python -m tide_lite.cli.aggregate_cli
> python -m tide_lite.cli.plots_cli
> ```

---

"""
            # Insert after executive summary
            exec_summary_end = template.find("---", template.find("## Executive Summary"))
            if exec_summary_end != -1:
                template = template[:exec_summary_end] + todo_section + template[exec_summary_end:]
                
        return template
        
    def generate_report(self):
        """Main method to generate the report."""
        logger.info("Starting report generation...")
        
        # Load template
        template = self.load_template()
        logger.info(f"Loaded template from {self.template_path}")
        
        # Load summary results
        summary = self.load_summary()
        if summary:
            logger.info("Loaded experimental results")
        else:
            logger.warning("No experimental results found - generating template with TODOs")
            
        # Fill in the template
        report = template
        report = self.fill_results_table(report, summary)
        report = self.update_plots(report)
        report = self.update_metadata(report)
        report = self.add_todos(report, summary)
        
        # Save the report
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write(report)
            
        logger.info(f"Report generated successfully: {self.output_path}")
        
        # Print summary statistics if available
        if summary:
            print("\n" + "="*50)
            print("SUMMARY STATISTICS")
            print("="*50)
            
            best_method = summary.get("best_method", "Unknown")
            best_metric = summary.get("best_metric_value", 0)
            
            print(f"Best Method: {best_method}")
            print(f"Best Performance: {best_metric:.4f}")
            print("="*50)
        else:
            print("\n⚠️  No results to summarize - run experiments first!")
            

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate TIDE-Lite report from experimental results"
    )
    
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("reports/report_template.md"),
        help="Path to the report template (default: reports/report_template.md)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/report.md"),
        help="Output path for generated report (default: reports/report.md)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Generate the report
    generator = ReportGenerator(args.template, args.output)
    
    try:
        generator.generate_report()
        print(f"\n✅ Report successfully generated: {args.output}")
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise
        

if __name__ == "__main__":
    main()
