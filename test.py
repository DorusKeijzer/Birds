import torch
import sys

def inspect_model(model_path):
    try:
        # Try to load the entire model
        model = torch.load(model_path)
        print("Successfully loaded the entire model.")
        print("Type of loaded object:", type(model))
        if isinstance(model, dict):
            print("Keys in the model dict:", model.keys())
    except Exception as e:
        print(f"Error loading entire model: {e}")

    try:
        # Try to load just the state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        print("\nSuccessfully loaded the state dict.")
        print("Keys in the state dict:")
        for key in state_dict.keys():
            print(f"- {key}")
    except Exception as e:
        print(f"Error loading state dict: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_model.py <path_to_model>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    inspect_model(model_path)

