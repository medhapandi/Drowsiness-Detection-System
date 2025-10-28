# install_requirements.py
import subprocess
import sys

def install_packages():
    packages = [
        'opencv-python==4.8.1',
        'tensorflow==2.13.0',
        'numpy==1.24.3',
        'keras==2.13.1',
        'pygame==2.5.0',
        'scikit-learn==1.3.0',
        'imutils==0.5.4',
        'dlib==19.24.0'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Successfully installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("\nAll packages installed successfully!")

if __name__ == "__main__":
    install_packages()