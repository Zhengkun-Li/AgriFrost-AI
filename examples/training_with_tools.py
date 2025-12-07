"""
Complete end-to-end example: Training with all model utils tools.

This example demonstrates how to use all improved training tools together:
- ProgressLogger: Logging with rotation and flush optimization
- TrainingHistory: History tracking with unified fields
- CheckpointManager: GPU/CPU compatible checkpoints with best-k saving
- ConfigValidator: 2×2+1 framework validation
- GraphBuilder: Graph construction with metadata export
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.utils import ProgressLogger, TrainingHistory, CheckpointManager
from src.config.schema.validator import ConfigValidator
from src.models.utils.graph_builder import GraphBuilder
from src.utils.metadata import ExperimentMetadata


def main():
    """Main training workflow example."""
    
    # Setup output directory
    output_dir = Path("examples/output/training_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Validate configuration (2×2+1 framework)
    print("=" * 70)
    print("Step 1: Validate Configuration (2×2+1 Framework)")
    print("=" * 70)
    
    valid, msg = ConfigValidator.validate_experiment_metadata(
        matrix_cell='C',
        track='raw',
        horizon_h=12,
        model_name='dcrnn',
        radius_km=50.0
    )
    
    if not valid:
        raise ValueError(f"Invalid configuration: {msg}")
    
    print(f"✅ Configuration validated: {msg or 'OK'}")
    
    # Step 2: Setup tools
    print("\n" + "=" * 70)
    print("Step 2: Setup Training Tools")
    print("=" * 70)
    
    # ProgressLogger with optimized settings (event-based API)
    logger = ProgressLogger(
        flush_interval=10,  # Flush every 10 messages (optimized)
        max_log_size_mb=10.0,  # Rotate at 10MB
        use_metric_schema=True
    )
    # Bind log files (path management)
    logger.bind_files(
        brief_path=output_dir / "training.log",
        detailed_path=output_dir / "training_detailed.log"
    )
    
    # TrainingHistory with unified fields (schema injection)
    history = TrainingHistory(
        metrics=['train_loss', 'val_loss', 'learning_rate', 'epoch_time'],
        use_metric_schema=True
    )
    
    # CheckpointManager with best-k support (path management)
    checkpoint_mgr = CheckpointManager(
        checkpoint_frequency=5,  # Save every 5 epochs
        save_best=True,
        best_metric="val_loss",
        keep_top_k=3  # Keep top 3 checkpoints
    )
    checkpoint_mgr.bind_dir(output_dir / "checkpoints")
    
    print("✅ All tools initialized")
    
    # Step 3: Create ExperimentMetadata (2×2+1 framework)
    print("\n" + "=" * 70)
    print("Step 3: Create Experiment Metadata")
    print("=" * 70)
    
    experiment_metadata = ExperimentMetadata(
        matrix_cell='C',
        track='raw',
        model_name='dcrnn',
        horizon_h=12,
        radius_km=50.0,
        training_scope='full_training'
    )
    experiment_metadata.save(output_dir)
    
    print(f"✅ Experiment metadata saved: {output_dir / 'run_metadata.json'}")
    
    # Step 4: Build graph (if using graph models)
    print("\n" + "=" * 70)
    print("Step 4: Build Graph Structure")
    print("=" * 70)
    
    # Note: This requires actual station metadata file
    # For example purposes, we'll skip if file doesn't exist
    metadata_path = project_root / "data" / "external" / "cimis_station_metadata.json"
    
    if metadata_path.exists():
        builder = GraphBuilder(metadata_path=metadata_path)
        graph = builder.build_radius_graph(radius_km=50.0)
        
        # Save graph with metadata export (2×2+1 compatibility)
        GraphBuilder.save_graph(
            graph,
            output_dir / "graph.pkl",
            metadata_path=output_dir / "run_metadata.json"
        )
        
        print(f"✅ Graph built and saved with metadata export")
        print(f"   Graph type: {graph['graph_type']}")
        print(f"   Graph param: {graph['graph_param']}")
    else:
        print(f"⚠️  Station metadata not found: {metadata_path}")
        print("   Skipping graph building (requires actual metadata file)")
    
    # Step 5: Training loop
    print("\n" + "=" * 70)
    print("Step 5: Training Loop (Simulated)")
    print("=" * 70)
    
    # Use event-based API
    logger.on_training_start("DCRNN", device="cpu", config={"batch_size": 32, "epochs": 10})
    
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # Simulate training
        train_loss = 0.5 - epoch * 0.02
        val_loss = 0.45 - epoch * 0.015
        learning_rate = 0.001 * (0.9 ** epoch)
        epoch_time = 10.0 + epoch * 0.1
        
        # Log epoch with unified fields (event-based API)
        logger.on_epoch(
            epoch=epoch,
            total_epochs=num_epochs,
            metrics={
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': learning_rate,
                'epoch_time': epoch_time
            },
            use_tqdm=False
        )
        
        # Record in history with unified fields
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
                model_state={'weight': [1.0, 2.0, 3.0]},  # Simulated
                optimizer_state={'lr': learning_rate},
                metrics={'train_loss': train_loss, 'val_loss': val_loss}
            )
            logger.log(f"  💾 Checkpoint saved at epoch {epoch}", flush=True)
        
        # Save best model
        if val_loss < best_val_loss:
            if checkpoint_mgr.save_best_checkpoint(
                epoch=epoch,
                model_state={'weight': [1.0, 2.0, 3.0]},
                metric_value=val_loss
            ):
                logger.on_metric_improved("val_loss", val_loss, best_val_loss, epoch=epoch)
                history.set_best('val_loss', val_loss, epoch)
                best_val_loss = val_loss
    
    # Step 6: Save training artifacts
    print("\n" + "=" * 70)
    print("Step 6: Save Training Artifacts")
    print("=" * 70)
    
    # Save history
    history.save(output_dir / "training_history.json")
    print(f"✅ Training history saved: {output_dir / 'training_history.json'}")
    
    # Save training curves (stateless function)
    from src.visualization.plots import plot_training_curves
    curve_path = output_dir / "curves" / "training_curves.png"
    plot_training_curves(
        history=history,  # Direct TrainingHistory instance
        save_path=curve_path,
        title="Training Curves"
    )
    print(f"✅ Training curves saved")
    
    # Complete training
    logger.on_training_complete(
        total_time=sum(history.epoch_times) if history.epoch_times else 100.0,
        total_epochs=num_epochs
    )
    
    # Step 7: Resume training example
    print("\n" + "=" * 70)
    print("Step 7: Resume Training Example")
    print("=" * 70)
    
    resume_info = checkpoint_mgr.resume_training(epoch=5)
    if resume_info:
        print(f"✅ Resumed from epoch {resume_info['epoch']}")
        print(f"   Model state: {list(resume_info['model_state'].keys())[:3]}...")
        print(f"   Optimizer state: {list(resume_info['optimizer_state'].keys())[:3]}...")
        print(f"   Metrics: {resume_info.get('metrics', {})}")
    else:
        print("⚠️  No checkpoint found for resume")
    
    print("\n" + "=" * 70)
    print("✅ Training Example Complete")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {output_dir / 'run_metadata.json'}")
    print(f"  - {output_dir / 'training.log'}")
    print(f"  - {output_dir / 'training_detailed.log'}")
    print(f"  - {output_dir / 'training_history.json'}")
    print(f"  - {output_dir / 'checkpoints/'}")


if __name__ == "__main__":
    main()

