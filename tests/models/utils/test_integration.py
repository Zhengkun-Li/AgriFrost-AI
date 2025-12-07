"""Integration tests for model utils tools."""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.models.utils.progress_logger import ProgressLogger
from src.models.utils.training_history import TrainingHistory
from src.models.utils.checkpoint_manager import CheckpointManager
from src.config.schema.validator import ConfigValidator
from src.models.utils.graph_builder import GraphBuilder
from src.utils.metadata import ExperimentMetadata


class TestIntegration:
    """Integration tests for tool interactions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_progress_logger_training_history_field_unification(self, temp_dir):
        """Test ProgressLogger and TrainingHistory field unification."""
        # Setup
        logger = ProgressLogger(use_metric_schema=True)
        logger.bind_files(
            brief_path=temp_dir / "training.log",
            detailed_path=temp_dir / "training_detailed.log"
        )
        history = TrainingHistory(
            metrics=['train_loss', 'val_loss', 'learning_rate', 'epoch_time'],
            use_metric_schema=True
        )
        
        # Record epoch with unified fields
        epoch = 1
        train_loss = 0.5
        val_loss = 0.45
        learning_rate = 0.01
        epoch_time = 12.5
        
        # Both use same field names (event-based API)
        logger.on_epoch(
            epoch=epoch,
            total_epochs=100,
            metrics={
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': learning_rate,
                'epoch_time': epoch_time
            },
            use_tqdm=False
        )
        
        history.record_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            epoch_time=epoch_time
        )
        
        # Verify consistency
        assert history.history['train_loss'][0] == train_loss
        assert history.history['val_loss'][0] == val_loss
        assert history.history['learning_rate'][0] == learning_rate
        assert history.history['epoch_time'][0] == epoch_time
        
        # Verify log contains unified field format
        log_content = (temp_dir / "training_detailed.log").read_text(encoding='utf-8')
        assert "train_loss=0.500000" in log_content
        assert "val_loss=0.450000" in log_content
    
    def test_graph_builder_metadata_export(self, temp_dir):
        """Test GraphBuilder metadata export to run_metadata.json."""
        # Create sample metadata
        metadata_path = temp_dir / "station_metadata.json"
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump([
                {"Stn Id": 1, "Latitude": 38.5, "Longitude": -121.5},
                {"Stn Id": 2, "Latitude": 38.6, "Longitude": -121.6}
            ], f)
        
        # Create ExperimentMetadata
        experiment_metadata = ExperimentMetadata(
            matrix_cell='C',
            track='raw',
            model_name='dcrnn',
            horizon_h=12
        )
        run_metadata_path = temp_dir / "run_metadata.json"
        experiment_metadata.save(temp_dir)
        
        # Build graph
        builder = GraphBuilder(metadata_path=metadata_path)
        graph = builder.build_radius_graph(radius_km=50.0)
        
        # Save with metadata export
        GraphBuilder.save_graph(graph, temp_dir / "graph.pkl", metadata_path=run_metadata_path)
        
        # Verify metadata updated
        loaded_metadata = ExperimentMetadata.load(run_metadata_path)
        assert loaded_metadata.radius_km == 50.0
        assert loaded_metadata.matrix_cell == 'C'
        assert loaded_metadata.model_name == 'dcrnn'
    
    def test_config_validator_2x2_plus_1_rules(self):
        """Test ConfigValidator 2×2+1 framework rules."""
        # Test C cell: must have radius
        valid, msg = ConfigValidator.validate_experiment_metadata(
            matrix_cell='C',
            track='raw',
            horizon_h=12,
            radius_km=50.0
        )
        assert valid is True
        
        # Test C cell: missing radius (should fail)
        valid, msg = ConfigValidator.validate_experiment_metadata(
            matrix_cell='C',
            track='raw',
            horizon_h=12,
            radius_km=None
        )
        assert valid is False
        assert 'must have radius_km' in msg.lower()
        
        # Test E cell: must have knn_k
        valid, msg = ConfigValidator.validate_experiment_metadata(
            matrix_cell='E',
            track='raw',
            horizon_h=12,
            knn_k=5
        )
        assert valid is True
        
        # Test E cell: missing knn_k (should fail)
        valid, msg = ConfigValidator.validate_experiment_metadata(
            matrix_cell='E',
            track='raw',
            horizon_h=12,
            knn_k=None
        )
        assert valid is False
        assert 'must have knn_k' in msg.lower()
        
        # Test A cell: cannot have radius
        valid, msg = ConfigValidator.validate_experiment_metadata(
            matrix_cell='A',
            track='raw',
            horizon_h=12,
            radius_km=50.0
        )
        assert valid is False
        assert 'cannot have radius_km' in msg.lower() or 'no radius' in msg.lower()
    
    def test_checkpoint_manager_resume_training(self, temp_dir):
        """Test CheckpointManager resume training functionality."""
        checkpoint_mgr = CheckpointManager(
            checkpoint_frequency=10,
            save_best=True,
            best_metric="val_loss",
            keep_top_k=3
        )
        checkpoint_mgr.bind_dir(temp_dir / "checkpoints")
        
        # Simulate training and save checkpoint
        model_state = {'layer1.weight': [1.0, 2.0, 3.0], 'layer1.bias': [0.5]}
        optimizer_state = {'param_groups': [{'lr': 0.001}]}
        scheduler_state = {'step': 5}
        
        checkpoint_mgr.save_checkpoint(
            epoch=10,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            metrics={'val_loss': 0.45}
        )
        
        # Resume training
        resume_info = checkpoint_mgr.resume_training(epoch=10)
        
        assert resume_info is not None
        assert resume_info['epoch'] == 10
        assert resume_info['model_state'] == model_state
        assert resume_info['optimizer_state'] == optimizer_state
        assert resume_info['scheduler_state'] == scheduler_state
        assert resume_info['metrics']['val_loss'] == 0.45
        
        # Verify internal state updated
        assert checkpoint_mgr.best_epoch == 10
    
    def test_full_training_workflow(self, temp_dir):
        """Test full training workflow with all tools."""
        # Setup tools
        logger = ProgressLogger(use_metric_schema=True)
        logger.bind_files(
            brief_path=temp_dir / "training.log",
            detailed_path=temp_dir / "training_detailed.log"
        )
        history = TrainingHistory(
            metrics=['train_loss', 'val_loss', 'learning_rate', 'epoch_time'],
            use_metric_schema=True
        )
        checkpoint_mgr = CheckpointManager(
            checkpoint_frequency=5,
            save_best=True,
            best_metric="val_loss"
        )
        checkpoint_mgr.bind_dir(temp_dir / "checkpoints")
        
        # Simulate training loop
        logger.on_training_start("LSTM", device="cpu", config={"epochs": 10})
        
        for epoch in range(1, 11):
            train_loss = 0.5 - epoch * 0.01
            val_loss = 0.45 - epoch * 0.008
            learning_rate = 0.001 * (0.9 ** epoch)
            epoch_time = 10.0 + epoch * 0.1
            
            # Log epoch (unified fields, event-based API)
            logger.on_epoch(
                epoch=epoch,
                total_epochs=10,
                metrics={
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': learning_rate,
                    'epoch_time': epoch_time
                },
                use_tqdm=False
            )
            
            # Record in history (unified fields)
            history.record_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=learning_rate,
                epoch_time=epoch_time
            )
            
            # Save checkpoint if needed
            if checkpoint_mgr.should_save_checkpoint(epoch):
                checkpoint_mgr.save_checkpoint(
                    epoch=epoch,
                    model_state={'weight': [1.0, 2.0, 3.0]},
                    metrics={'val_loss': val_loss}
                )
            
            # Save best
            checkpoint_mgr.save_best_checkpoint(
                epoch=epoch,
                model_state={'weight': [1.0, 2.0, 3.0]},
                metric_value=val_loss
            )
        
        # Verify history
        assert len(history) == 10
        assert len(history.history['train_loss']) == 10
        assert len(history.history['val_loss']) == 10
        assert len(history.history['epoch_time']) == 10
        
        # Verify checkpoints
        checkpoints = checkpoint_mgr.list_checkpoints()
        assert len(checkpoints) >= 2  # At least periodic + best
        
        # Verify log files
        assert (temp_dir / "training.log").exists()
        assert (temp_dir / "training_detailed.log").exists()
        
        # Save history
        history.save(temp_dir / "training_history.json")
        assert (temp_dir / "training_history.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

