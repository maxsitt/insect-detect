<<<<<<< HEAD
# NIGHTLIFE

<img src="https://github.com/darasafe/nightlife/assets/103866780/931084f5-e549-4565-adad-25c9ae236fcd" height="200">

Install the required dependencies for Raspberry Pi + OAK by running:

```bash
sudo curl -fL https://docs.luxonis.com/install_dependencies.sh | bash
```

Install the package libopenblas-dev (required for latest numpy version):

```bash
sudo apt-get install libopenblas-dev
```

To ensure you have all necessary libraries and dependencies installed in the Raspberry Pi, please download the `requirements.txt` file from the repository and install it using pip:

```bash
python3 -m pip install -r insect-detect/requirements.txt

This project builds on "Insect Detect" by Max Sittinger et al., 2023, innovating on their open-source camera trap for automated insect monitoring.

Sittinger, M., Uhler, J., Pink, M., & Herz, A. (2023). Insect Detect: An Open-Source DIY Camera Trap for Automated Insect Monitoring [Preprint]. bioRxiv. https://doi.org/10.1101/2023.12.05.570242