import os
import mlflow
from typing import Any

def setup_mlflow_run(experiment_name: str, project_root: str) -> Any:
    """Sets up an isolated, resume-safe MLflow run tracking environment.
    
    1. Sets tracking URI to an isolated local path: runs/<experiment_name>/mlruns/
    2. Sets the experiment name to <experiment_name>.
    3. Checks for runs/<experiment_name>/run_id.txt:
       - If found, resumes that specific MLflow run.
       - If not found, starts a new run and writes its ID to run_id.txt.
    """
    # Allow local filesystem tracking backend
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    # Create the isolated experiment run directory
    run_dir = os.path.join(project_root, "runs", experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Set the tracking URI to the isolated subdirectory
    mlruns_dir = os.path.join(run_dir, "mlruns")
    tracking_uri = f"file://{os.path.abspath(mlruns_dir)}"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Establish the MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Check for saved run_id to support resumability
    run_id_file = os.path.join(run_dir, "run_id.txt")
    if os.path.exists(run_id_file):
        with open(run_id_file, "r", encoding="utf-8") as f:
            run_id = f.read().strip()
        print(f"Found active run ID. Resuming MLflow run: '{run_id}'")
        try:
            run = mlflow.start_run(run_id=run_id)
        except Exception as e:
            print(f"Could not resume MLflow run '{run_id}' (it may be finished or invalid): {e}")
            print("Starting a new MLflow run instead...")
            run = mlflow.start_run()
            run_id = run.info.run_id
            with open(run_id_file, "w", encoding="utf-8") as f:
                f.write(run_id)
            print(f"New MLflow run started and logged: '{run_id}'")
    else:
        print("No active run ID found. Starting a new MLflow run...")
        run = mlflow.start_run()
        run_id = run.info.run_id
        with open(run_id_file, "w", encoding="utf-8") as f:
            f.write(run_id)
        print(f"New MLflow run started and logged: '{run_id}'")
        
    return run
