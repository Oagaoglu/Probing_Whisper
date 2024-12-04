
#Probing

This repository contains code for encoding extraction, voice activity detection (VAD), and acoustic feature extraction using OpenSMILE, configured to align with Whisper's frame and stride specifications. Also the probing code in per feature setting and in vector.

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Install Dependencies**
   Use the provided `environment.yml` to create a Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate <env_name>
   ```

---

## How to Use

### 1. Encoding Extraction
Run `main.py` to extract encodings:
```bash
python main.py
```

### 2. Voice Activity Detection (VAD)
Generate a CSV with boolean values for voice activity detection by running:
```bash
python vad_csv.py
```

### 3. Acoustic Feature Extraction
Extract acoustic features using `f_extractor.py`. Before running, modify the configuration file at `[wherever conda installed opensmile]/config/gemaps/v01b/GeMAPSv01b_core.lld.conf.inc`:

#### Configuration Changes:
Update the following objects:

- **Object: `[gemapsv01b_frame25:cFramer]`**
  ```ini
  frameSize = 0.020
  frameStep = 0.010
  ```

- **Object: `[gemapsv01b_frame60:cFramer]`**
  ```ini
  frameSize = 0.020
  frameStep = 0.010
  ```

These settings ensure compatibility with Whisper's 20ms frames and 10ms stride.

Run the script:
```bash
python f_extractor.py
```

---

## Notes
- Ensure OpenSMILE is installed and accessible in your environment.
- Verify changes to the configuration file to avoid syntax errors.
- Modify the dataset location and in general the directories for csv files.

---

## Contact
For any question you can reach me through orhanagaoglu2002@gmail.com
