import json
import numpy as np

# Function to generate a single entry
def generate_entry():
    # Random position (x, y, z)
    x, y, z = np.random.uniform(-1, 1, 3)
    
    # Random orientation (theta)
    theta = np.random.uniform(0, 360)
    
    # Random grasp score
    grasp_score = np.random.uniform(0, 1)
    
    # Random image (300x300x3, pixel values between 0 and 255)
    image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8).tolist()
    
    # Create the entry as a dictionary
    entry = {
        "x": x,
        "y": y,
        "z": z,
        "theta": theta,
        "image": image,
        "grasp_score": grasp_score
    }
    
    return entry

# Generate the dummy dataset (e.g., 100 entries)
dummy_data = [generate_entry() for _ in range(100)]

# Save the dataset to a JSON file
with open('dummy_data.json', 'w') as f:
    json.dump(dummy_data, f, indent=4)

print("Dummy dataset has been generated and saved to 'dummy_data.json'.")
