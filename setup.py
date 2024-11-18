import os
import subprocess
import sys

def create_virtualenv():
    """Creates a virtual environment in the `.venv` folder."""
    venv_dir = ".venv"
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        print(f"Virtual environment created at {venv_dir}")
    else:
        print(f"Virtual environment already exists at {venv_dir}")

def install_requirements():
    """Installs requirements from a requirements.txt file."""
    requirements = [
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "pandas",
        "scikit-learn",
        "rouge-score",
        "nltk",
        "detoxify",
        "lm-eval",
        "matplotlib"
    ]
    venv_bin = os.path.join(".venv", "bin" if os.name != "nt" else "Scripts")
    pip_path = os.path.join(venv_bin, "pip")
    
    print("Installing requirements...")
    subprocess.check_call([pip_path, "install", "--index-url", "https://download.pytorch.org/whl/cu118"] + requirements)
    print("All requirements installed.")

def main():
    print("Setting up Python environment...")
    create_virtualenv()
    install_requirements()
    print("Setup complete. Activate the virtual environment using:")
    print(f"source .venv/bin/activate" if os.name != "nt" else ".venv\\Scripts\\activate")

if __name__ == "__main__":
    main()