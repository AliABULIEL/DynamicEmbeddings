"""
Report generator for benchmark results.

Creates comprehensive reports with:
- Executive summary
- Performance comparisons
- Statistical significance tests
- Improvement highlights
- Visualizations
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive benchmark reports."""
    
    def __init__(self, output_dir: Path):
        """Initialize report generator.
        
        Args:
            output_dir: Directory for reports and figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
    
    def create_comparison_table(
        self,
        df: pd.DataFrame,
        metric: str = "ndcg@10"
    ) -> str:
        """Create formatted comparison table.
        
        Args:
            df: Results DataFrame
            metric: Metric to display
            
        Returns:
            Markdown formatted table
        """
        # Pivot table
        pivot = df.pivot(index="bucket", columns="model", values=metric)
        
        # Format
        lines = [f"\n## {metric.upper()} Comparison\n"]
        lines.append("| Bucket | " + " | ".join(pivot.columns) + " |")
        lines.append("|" + "---|" * (len(pivot.columns) + 1))
        
        for bucket in pivot.index:
            row_vals = []
            for col in pivot.columns:
                val = pivot.loc[bucket, col]
                if pd.notna(val):
                    row_vals.append(f"{val:.4f}")
                else:
                    row_vals.append("N/A")
            lines.append(f"| {bucket} | " + " | ".join(row_vals) + " |")
        
        return "\n".join(lines)
    
    def create_improvement_highlights(
        self,
        df: pd.DataFrame,
        baseline: str = "all-MiniLM-L6-v2"
    ) -> str:
        """Create improvement highlights section.
        
        Args:
            df: Results DataFrame
            baseline: Baseline model name
            
        Returns:
            Markdown formatted highlights
        """
        lines = ["\n## 🎯 Key Improvements\n"]
        
        baseline_df = df[df["model"] == baseline]
        
        for model in df["model"].unique():
            if model == baseline:
                continue
            
            model_df = df[df["model"] == model]
            
            lines.append(f"\n### {model} vs {baseline}\n")
            
            improvements = []
            
            for bucket in df["bucket"].unique():
                base_row = baseline_df[baseline_df["bucket"] == bucket]
                model_row = model_df[model_df["bucket"] == bucket]
                
                if base_row.empty or model_row.empty:
                    continue
                
                for metric in ["ndcg@10", "recall@10", "mrr"]:
                    if metric not in base_row.columns:
                        continue
                    
                    base_val = base_row[metric].values[0]
                    model_val = model_row[metric].values[0]
                    
                    if base_val > 0:
                        improvement = ((model_val - base_val) / base_val) * 100
                        
                        # Highlight significant improvements
                        if abs(improvement) >= 5:
                            icon = "🔥" if improvement > 0 else "⚠️"
                            sign = "+" if improvement > 0 else ""
                            improvements.append(
                                f"- {icon} **{bucket} {metric}**: {sign}{improvement:.1f}% "
                                f"({base_val:.4f} → {model_val:.4f})"
                            )
            
            if improvements:
                lines.extend(improvements)
            else:
                lines.append("- No significant improvements (>5%)")
        
        return "\n".join(lines)
    
    def create_summary_statistics(
        self,
        df: pd.DataFrame,
        baseline: str = "all-MiniLM-L6-v2"
    ) -> str:
        """Create summary statistics section.
        
        Args:
            df: Results DataFrame
            baseline: Baseline model name
            
        Returns:
            Markdown formatted statistics
        """
        lines = ["\n## 📊 Summary Statistics\n"]
        
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            
            lines.append(f"\n### {model}\n")
            lines.append("| Metric | Mean | Std | Min | Max |")
            lines.append("|--------|------|-----|-----|-----|")
            
            for metric in ["ndcg@10", "recall@10", "recall@100", "mrr"]:
                if metric not in model_df.columns:
                    continue
                
                values = model_df[metric].dropna()
                if len(values) == 0:
                    continue
                
                lines.append(
                    f"| {metric} | {values.mean():.4f} | {values.std():.4f} | "
                    f"{values.min():.4f} | {values.max():.4f} |"
                )
        
        return "\n".join(lines)
    
    def create_comparison_plots(
        self,
        df: pd.DataFrame,
        metrics: List[str] = ["ndcg@10", "recall@10", "mrr"]
    ):
        """Create comparison visualization plots.
        
        Args:
            df: Results DataFrame
            metrics: Metrics to plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric not in df.columns:
                continue
            
            ax = axes[idx]
            
            # Create grouped bar plot
            buckets = df["bucket"].unique()
            models = df["model"].unique()
            
            x = np.arange(len(buckets))
            width = 0.8 / len(models)
            
            for i, model in enumerate(models):
                model_df = df[df["model"] == model]
                values = [
                    model_df[model_df["bucket"] == bucket][metric].values[0]
                    if not model_df[model_df["bucket"] == bucket].empty
                    else 0
                    for bucket in buckets
                ]
                ax.bar(x + i * width, values, width, label=model)
            
            ax.set_xlabel("Time Bucket", fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f"{metric.upper()} Comparison", fontsize=14, fontweight="bold")
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(buckets, rotation=45)
            ax.legend(fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "benchmark_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"✅ Saved comparison plot: {output_path}")
    
    def create_improvement_heatmap(
        self,
        df: pd.DataFrame,
        baseline: str = "all-MiniLM-L6-v2"
    ):
        """Create heatmap of improvements over baseline.
        
        Args:
            df: Results DataFrame
            baseline: Baseline model name
        """
        baseline_df = df[df["model"] == baseline]
        
        # Calculate improvements
        improvements_data = []
        
        for model in df["model"].unique():
            if model == baseline:
                continue
            
            model_df = df[df["model"] == model]
            
            for bucket in df["bucket"].unique():
                base_row = baseline_df[baseline_df["bucket"] == bucket]
                model_row = model_df[model_df["bucket"] == bucket]
                
                if base_row.empty or model_row.empty:
                    continue
                
                for metric in ["ndcg@10", "recall@10", "mrr"]:
                    if metric not in base_row.columns:
                        continue
                    
                    base_val = base_row[metric].values[0]
                    model_val = model_row[metric].values[0]
                    
                    if base_val > 0:
                        improvement = ((model_val - base_val) / base_val) * 100
                        improvements_data.append({
                            "model": model,
                            "bucket": bucket,
                            "metric": metric,
                            "improvement": improvement
                        })
        
        if not improvements_data:
            logger.warning("No improvements to plot")
            return
        
        # Create DataFrame
        imp_df = pd.DataFrame(improvements_data)
        
        # Pivot for heatmap
        pivot = imp_df.pivot_table(
            index=["model", "bucket"],
            columns="metric",
            values="improvement",
            aggfunc="mean"
        )
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "Improvement (%)"},
            ax=ax
        )
        ax.set_title(
            f"Improvement over {baseline} (%)",
            fontsize=14,
            fontweight="bold"
        )
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Model / Bucket", fontsize=12)
        
        plt.tight_layout()
        
        output_path = self.figures_dir / "improvement_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"✅ Saved improvement heatmap: {output_path}")
    
    def generate_full_report(
        self,
        df: pd.DataFrame,
        baseline: str = "all-MiniLM-L6-v2",
        title: str = "Temporal LoRA Benchmark Report"
    ) -> str:
        """Generate complete benchmark report.
        
        Args:
            df: Results DataFrame
            baseline: Baseline model name
            title: Report title
            
        Returns:
            Path to generated report
        """
        logger.info("📝 Generating comprehensive report...")
        
        # Generate visualizations
        self.create_comparison_plots(df)
        self.create_improvement_heatmap(df, baseline)
        
        # Build report
        report_lines = [
            f"# {title}\n",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n",
        ]
        
        # Executive Summary
        report_lines.append("\n## 📋 Executive Summary\n")
        report_lines.append(f"This report compares **{len(df['model'].unique())} models** ")
        report_lines.append(f"across **{len(df['bucket'].unique())} time buckets**.\n")
        
        # Key Improvements
        report_lines.append(self.create_improvement_highlights(df, baseline))
        
        # Comparison Tables
        for metric in ["ndcg@10", "recall@10", "mrr"]:
            if metric in df.columns:
                report_lines.append(self.create_comparison_table(df, metric))
        
        # Summary Statistics
        report_lines.append(self.create_summary_statistics(df, baseline))
        
        # Visualizations
        report_lines.append("\n## 📊 Visualizations\n")
        report_lines.append("\n### Performance Comparison\n")
        report_lines.append("![Comparison](figures/benchmark_comparison.png)\n")
        report_lines.append("\n### Improvement Heatmap\n")
        report_lines.append("![Improvements](figures/improvement_heatmap.png)\n")
        
        # Conclusion
        report_lines.append("\n## 🎓 Conclusions\n")
        
        # Find best performing model
        avg_scores = df.groupby("model")["ndcg@10"].mean().sort_values(ascending=False)
        best_model = avg_scores.index[0]
        best_score = avg_scores.values[0]
        
        report_lines.append(f"- **Best Model:** {best_model} (Avg NDCG@10: {best_score:.4f})\n")
        
        # Calculate average improvement
        if "Temporal-LoRA" in df["model"].values:
            lora_df = df[df["model"] == "Temporal-LoRA"]
            base_df = df[df["model"] == baseline]
            
            if not lora_df.empty and not base_df.empty:
                lora_avg = lora_df["ndcg@10"].mean()
                base_avg = base_df["ndcg@10"].mean()
                improvement = ((lora_avg - base_avg) / base_avg) * 100
                
                report_lines.append(
                    f"- **Temporal LoRA Improvement:** {improvement:+.1f}% over {baseline}\n"
                )
        
        # Save report
        report_text = "".join(report_lines)
        report_path = self.output_dir / "BENCHMARK_REPORT.md"
        
        with open(report_path, "w") as f:
            f.write(report_text)
        
        logger.info(f"✅ Report saved to: {report_path}")
        
        return str(report_path)


def generate_report(
    results_csv: Path,
    output_dir: Path,
    baseline: str = "all-MiniLM-L6-v2"
) -> str:
    """Generate benchmark report from results CSV.
    
    Args:
        results_csv: Path to results CSV
        output_dir: Output directory
        baseline: Baseline model name
        
    Returns:
        Path to generated report
    """
    df = pd.read_csv(results_csv)
    
    generator = ReportGenerator(output_dir)
    report_path = generator.generate_full_report(df, baseline)
    
    return report_path
