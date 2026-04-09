import os
import subprocess

# List of all standard CIFAR-10-C corruptions
CORRUPTIONS = [
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "glass_blur",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "saturate",
    "shot_noise",
    "snow",
    "spatter",
    "speckle_noise",
    "zoom_blur"
]

def main():
    print("🚀 Starting sequential simulation across all CIFAR-10-C corruptions...")
    print(f"Total corruptions to evaluate: {len(CORRUPTIONS)}")
    
    for idx, corruption in enumerate(CORRUPTIONS, 1):
        print("\n" + "="*60)
        print(f"[{idx}/{len(CORRUPTIONS)}] ⏳ Running Simulation for: {corruption.upper()}")
        print("="*60)
        
        # We copy the environment variables and pass the new corruption dynamically
        env = os.environ.copy()
        env["CIFAR10C_CORRUPTION"] = corruption
        
        # Execute the main simulation 
        try:
            result = subprocess.run(
                ["python", "simulate_fl.py"], 
                env=env,
                check=True
            )
            print(f"✅ Finished {corruption} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error occurred while simulating {corruption}. Exit code: {e.returncode}")
            
    print("\n🎉 All sequential simulations completed!")

if __name__ == "__main__":
    main()
