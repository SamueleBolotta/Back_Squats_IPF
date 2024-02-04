# Back_Squats_IPF

## Project Overview

This project aims at creating a neural network that can assess the validity of a back squat. A valid back squat occurs when the crease of the hip is below the kneecap. The dataset was constructed with hundreds of images of powerlifting athletes - trying to achieve balance in terms of gender, skin colour and weight category. 

The labels are three: 
- "valid" when the IPF (International Powerlifting Federation) referees gave a white light to the lift from that specific perspective (i.e. frontal or semi-lateral)
- "parallel" when the IPF referees gave a red light to the lift from that specific perspective
- "above parallel" when the athlete was clearly above the parallel

 The Neural Network architecture utilizes a pre-trained ResNet18 model with a customized fully connected layer.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following prerequisites installed on your system:

    Python: This project requires Python 3.10.12. If you don't have it installed, download and install it from python.org.

    Other Dependencies: requirements.txt 

# Create a virtual environment 
    python3 -m venv your_env_name

# Activate the virtual environment
    source your_env_name/bin/activate 

# Navigate to your project directory
    cd path/to/your/project

# Clone repository
    git clone https://github.com/SamueleBolotta/Back_Squats_IPF.git

# Install project-specific dependencies (modify as needed)
    pip install -r requirements.txt

# Run the main file

    python3 main.py

## Contact
juan.camachomohedano@studenti.unitn.it
bolottasamuele@gmail.com



