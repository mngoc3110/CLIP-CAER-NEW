import os

def list_directory_contents(path, indent=0):
    """Recursively lists contents of a directory."""
    prefix = "  " * indent
    if not os.path.exists(path):
        print(f"{prefix}Directory not found: {path}")
        return

    print(f"{prefix}Contents of {path}:")
    try:
        items = os.listdir(path)
        if not items:
            print(f"{prefix}  (empty)")
        for item in sorted(items):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"{prefix}  - {item}/")
                list_directory_contents(item_path, indent + 1)
            else:
                print(f"{prefix}  - {item}")
    except Exception as e:
        print(f"{prefix}  Error listing contents: {e}")

base_path = "/kaggle/working/CLIP-CAER-NEW/"
debug_predictions_path = os.path.join(base_path, "debug_predictions")
debug_samples_path = os.path.join(base_path, "debug_samples")

print("Listing Debug Directories:")
print("="*30)

# List debug_predictions
list_directory_contents(debug_predictions_path)

print("\n" + "="*30)
# List debug_samples and its class subdirectories
list_directory_contents(debug_samples_path)

print("\n" + "="*30)
print("Listing specific class directories within debug_samples:")
for i in range(5): # Assuming class_0 to class_4 based on user's input
    class_path = os.path.join(debug_samples_path, f"class_{i}")
    list_directory_contents(class_path, indent=1)
    
print("\n" + "="*30)
print("Finished listing debug directories.")
