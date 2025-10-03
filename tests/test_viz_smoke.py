"""Smoke tests for visualization functions."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from temporal_lora.viz.plots import (
    plot_heatmaps_panel,
    plot_umap_sample,
    create_all_heatmaps,
)


class TestHeatmapPanel:
    """Test heatmap panel creation."""
    
    def test_plot_heatmaps_panel_basic(self):
        """Test basic heatmap panel creation."""
        # Create synthetic matrices
        buckets = ["bucket1", "bucket2", "bucket3"]
        
        baseline_matrix = pd.DataFrame(
            np.random.rand(3, 3),
            index=buckets,
            columns=buckets,
        )
        
        lora_matrix = pd.DataFrame(
            baseline_matrix.values + np.random.rand(3, 3) * 0.1,
            index=buckets,
            columns=buckets,
        )
        
        # Create heatmap panel
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "heatmap_panel.png"
            
            plot_heatmaps_panel(
                baseline_matrix=baseline_matrix,
                lora_matrix=lora_matrix,
                metric_name="NDCG@10",
                output_path=output_path,
            )
            
            # Check file was created
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_plot_heatmaps_panel_with_delta(self):
        """Test heatmap panel with explicit delta matrix."""
        buckets = ["bucket1", "bucket2"]
        
        baseline_matrix = pd.DataFrame(
            [[0.5, 0.3], [0.4, 0.6]],
            index=buckets,
            columns=buckets,
        )
        
        lora_matrix = pd.DataFrame(
            [[0.6, 0.4], [0.5, 0.7]],
            index=buckets,
            columns=buckets,
        )
        
        delta_matrix = lora_matrix - baseline_matrix
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "heatmap_with_delta.png"
            
            plot_heatmaps_panel(
                baseline_matrix=baseline_matrix,
                lora_matrix=lora_matrix,
                delta_matrix=delta_matrix,
                metric_name="Recall@10",
                output_path=output_path,
            )
            
            assert output_path.exists()
    
    def test_plot_heatmaps_panel_no_save(self):
        """Test heatmap panel without saving (display mode)."""
        buckets = ["bucket1", "bucket2"]
        
        baseline_matrix = pd.DataFrame(
            np.random.rand(2, 2),
            index=buckets,
            columns=buckets,
        )
        
        lora_matrix = pd.DataFrame(
            baseline_matrix.values + 0.1,
            index=buckets,
            columns=buckets,
        )
        
        # Should not raise error even without saving
        plot_heatmaps_panel(
            baseline_matrix=baseline_matrix,
            lora_matrix=lora_matrix,
            metric_name="MRR",
            output_path=None,
        )


class TestUMAPVisualization:
    """Test UMAP visualization."""
    
    def test_plot_umap_sample_basic(self):
        """Test basic UMAP visualization."""
        # Create synthetic embeddings
        embeddings_dict = {
            "bucket1": np.random.randn(100, 64).astype(np.float32),
            "bucket2": np.random.randn(100, 64).astype(np.float32),
            "bucket3": np.random.randn(100, 64).astype(np.float32),
        }
        
        # Normalize
        for bucket, emb in embeddings_dict.items():
            embeddings_dict[bucket] = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "umap.png"
            
            plot_umap_sample(
                embeddings_dict=embeddings_dict,
                output_path=output_path,
                max_points=200,
                seed=42,
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    def test_plot_umap_sample_with_sampling(self):
        """Test UMAP with sampling (more points than max_points)."""
        # Create large embeddings
        embeddings_dict = {
            "bucket1": np.random.randn(2000, 32).astype(np.float32),
            "bucket2": np.random.randn(2000, 32).astype(np.float32),
        }
        
        # Normalize
        for bucket, emb in embeddings_dict.items():
            embeddings_dict[bucket] = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "umap_sampled.png"
            
            plot_umap_sample(
                embeddings_dict=embeddings_dict,
                output_path=output_path,
                max_points=500,
                seed=42,
            )
            
            assert output_path.exists()
    
    def test_plot_umap_sample_reproducibility(self):
        """Test that UMAP visualization is reproducible with same seed."""
        embeddings_dict = {
            "bucket1": np.random.randn(50, 16).astype(np.float32),
            "bucket2": np.random.randn(50, 16).astype(np.float32),
        }
        
        # Normalize
        for bucket, emb in embeddings_dict.items():
            embeddings_dict[bucket] = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create two plots with same seed
            output1 = tmpdir / "umap1.png"
            output2 = tmpdir / "umap2.png"
            
            plot_umap_sample(
                embeddings_dict=embeddings_dict,
                output_path=output1,
                seed=42,
            )
            
            plot_umap_sample(
                embeddings_dict=embeddings_dict,
                output_path=output2,
                seed=42,
            )
            
            # Both should exist
            assert output1.exists()
            assert output2.exists()
            
            # Sizes should be similar (not exact due to matplotlib randomness)
            size1 = output1.stat().st_size
            size2 = output2.stat().st_size
            assert abs(size1 - size2) / size1 < 0.1  # Within 10%


class TestCreateAllHeatmaps:
    """Test batch heatmap creation."""
    
    def create_test_results(self, tmpdir: Path):
        """Create test result CSVs."""
        results_dir = tmpdir / "results"
        results_dir.mkdir()
        
        buckets = ["bucket1", "bucket2"]
        metrics = ["ndcg_at_10", "recall_at_10", "recall_at_100", "mrr"]
        
        # Create baseline and LoRA results
        for mode in ["baseline_frozen", "lora"]:
            for metric in metrics:
                df = pd.DataFrame(
                    np.random.rand(2, 2),
                    index=buckets,
                    columns=buckets,
                )
                
                csv_path = results_dir / f"{mode}_{metric}.csv"
                df.to_csv(csv_path)
        
        return results_dir
    
    def test_create_all_heatmaps(self):
        """Test creating all heatmaps at once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test data
            results_dir = self.create_test_results(tmpdir)
            output_dir = tmpdir / "figures"
            
            # Create all heatmaps
            create_all_heatmaps(
                results_dir=results_dir,
                output_dir=output_dir,
                baseline_mode="baseline_frozen",
                lora_mode="lora",
            )
            
            # Check that heatmaps were created
            expected_files = [
                "heatmap_panel_ndcg_at_10.png",
                "heatmap_panel_recall_at_10.png",
                "heatmap_panel_recall_at_100.png",
                "heatmap_panel_mrr.png",
            ]
            
            for filename in expected_files:
                filepath = output_dir / filename
                assert filepath.exists(), f"Missing: {filename}"
                assert filepath.stat().st_size > 0
    
    def test_create_all_heatmaps_missing_data(self):
        """Test graceful handling of missing data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create empty results directory
            results_dir = tmpdir / "results"
            results_dir.mkdir()
            
            output_dir = tmpdir / "figures"
            
            # Should not raise error
            create_all_heatmaps(
                results_dir=results_dir,
                output_dir=output_dir,
            )
            
            # Output dir should be created but no files
            assert output_dir.exists()


class TestEndToEndVisualization:
    """Test end-to-end visualization workflow."""
    
    def test_complete_visualization_workflow(self):
        """Test complete visualization workflow with synthetic data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create synthetic results
            results_dir = tmpdir / "results"
            results_dir.mkdir()
            
            buckets = ["early", "middle", "late"]
            
            # Create baseline and LoRA results
            for mode in ["baseline_frozen", "lora"]:
                for metric in ["ndcg_at_10", "recall_at_10"]:
                    matrix = pd.DataFrame(
                        np.random.rand(3, 3) * 0.8 + 0.1,  # 0.1 to 0.9
                        index=buckets,
                        columns=buckets,
                    )
                    
                    csv_path = results_dir / f"{mode}_{metric}.csv"
                    matrix.to_csv(csv_path)
            
            # Create synthetic embeddings
            embeddings_dir = tmpdir / "embeddings"
            
            for bucket in buckets:
                bucket_dir = embeddings_dir / bucket / "test"
                bucket_dir.mkdir(parents=True)
                
                # Create embeddings
                embeddings = np.random.randn(50, 32).astype(np.float32)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                
                np.save(bucket_dir / "embeddings.npy", embeddings)
                
                # Create IDs
                ids = [f"{bucket}_doc{i}" for i in range(50)]
                with open(bucket_dir / "ids.txt", "w") as f:
                    for doc_id in ids:
                        f.write(f"{doc_id}\n")
            
            # Run visualization
            from temporal_lora.viz.plots import visualize_results
            
            output_dir = tmpdir / "figures"
            
            visualize_results(
                results_dir=results_dir,
                embeddings_dir=embeddings_dir,
                output_dir=output_dir,
            )
            
            # Check outputs
            assert output_dir.exists()
            
            # Should have heatmaps
            assert (output_dir / "heatmap_panel_ndcg_at_10.png").exists()
            assert (output_dir / "heatmap_panel_recall_at_10.png").exists()
            
            # Should have UMAP
            assert (output_dir / "umap_embeddings.png").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
