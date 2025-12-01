import torch
from pathlib import Path

def inspect_checkpoint(model_name):
    path = Path(f'c:/Machine_Learning_Project/nfl_project_complete/nfl_project/models/{model_name}/best_{model_name}.pth')
    if not path.exists():
        print(f"{model_name}: Checkpoint not found")
        return

    try:
        checkpoint = torch.load(path, map_location='cpu')
        val_rmse = checkpoint.get('val_rmse', 'N/A')
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"{model_name.upper()}: Epoch {epoch}, Val RMSE: {val_rmse}")
    except Exception as e:
        print(f"{model_name}: Error loading checkpoint - {e}")

if __name__ == "__main__":
    print("Inspecting checkpoints...")
    inspect_checkpoint('lstm')
    inspect_checkpoint('gru')
    inspect_checkpoint('transformer')
