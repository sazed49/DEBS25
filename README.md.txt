

### MineDetect

#### Overview
MineDetect is a framework for implementing Robust Federated Learning (FL), specifically designed to detect and mitigate the impact of malicious and unreliable clients for both  a non-IID and IID federated learning setting. 

#### Features
- **Custom Dataset Loader**: Implements a `customloader.py` to handle datasets efficiently.

- **Federated Learning Server & Clients**:
  - `server.py`: Manages global model updates and client aggregation.
  - `clients.py`: Defines client-side training and interaction with the server.
- **Parsing & Logging**:
  - `parse.py`: Handles data parsing.
  - `logfiles.zip`: Contains results.
- **Automation Script**: `run.sh` simplifies execution.

#### Folder Structure
```
Robust_FL/
│── Dataset/           # Dataset files for training
│── models/            # Model architectures used for FL
│── rules/             # Contains foolsgold,krum,nuktikrum file
│── tools/             # Additional utilities
│── _main.py           # Entry point for FL implementation
│── clients.py         # Client object
│── customloader.py    # Data loading functionalities
│── logfiles.zip       # Results
│── main.py            # Main script for execution
│── malicious_clients.py # Malicious client class
│── parse.py           # Data parsing utilities
│── run.sh             # Shell script to run the framework
│── server.py          # Server-side FL aggregation
```
