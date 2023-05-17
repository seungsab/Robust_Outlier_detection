# Getting started for `Robust Outlier detection based on RPCA and Drift detection`

You are going to need a laptop or PC with Anaconda and several Python packages installed.
The following instruction would work as is for Mac or Ubuntu Linux users, Windows users would need to install and work in the [Git BASH](https://gitforwindows.org/) terminal.

**I would recommend strongly to use `Git` to use Linux commands even in Windows*

## Download and install Anaconda

Please go to the [Anaconda website](https://www.anaconda.com/).
Download and install *the latest* Anaconda version for latest *Python* for your operating system.


## Check-out the git repository of "'Robust Outlier detection"

Once Anaconda is ready, checkout the course repository and proceed with setting up the environment:

```bash
git clone https://github.com/seungsab/Robust_Outlier_detection.git
```

## Create isolated Anaconda environment

Change directory (`cd`) into the your folder
```bash
cd <your folder>
```

If you want to run this module, then type:
```bash
conda env create -f environment.yml
conda activate Robust_Outlier_detection
```
The current version of pytorch with the conda environment is Pytorch 1.7.1 (with CUDA 11.3 & cuDNN 8.7.0).



## Trouble shooting
#### **1. [Anaconda] Activating an environment fails**
```bash
$ conda activate Robust_Outlier_detection
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.
```
<!-- ![conda_init_error.PNG](./img/conda_init_error.PNG) -->

If you got the error message as above, simply change the command for activation under linux enviroment as:
```bash
source activate Robust_Outlier_detection
```

#### **2. [Errno 13] Permission denied:**
```bash
$ conda env create -f environment.yml
Collecting package metadata (repodata.json): done
Solving environment: done

Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
ERROR conda.core.link:_execute(745): An error occurred while installing package 'defaults::vs2015_runtime-14.27.29016-h5e58377_2'.
Rolling back transaction: done

[Errno 13] Permission denied: 'C:\\Users\\user\\anaconda3\\envs\\Robust_Outlier_detection\\vcruntime140.dll'
()
```

If you got the error message after failing installation like the above, you have to remove the subfolder of conda environment (herein, [folderName]) that was failed to install.
To do this, then type:

```bash
cd ~\anaconda3\envs
rm -rf [folderName]
```

If it doesn't work, please check the followings:
1) Check your path and typo
2) Check anything you missed to be described


#### **3. [SSL: CERTIFICATE_VERIFY_FAILED] **

```bash
$ conda env create -f environment.yml
Collecting package metadata (repodata.json): done
Solving environment: done

Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
Installing pip dependencies: - Ran pip subprocess with arguments:
['C:\\Users\\user\\anaconda3\\envs\\Robust_Outlier_detection\\python.exe', '-m', 'pip', 'install', '-U', '-r', 'C:\\Users\\user\\Desktop\\Robust_Outlier_detection\\condaenv.2o2y15x4.requirements.txt', '--exists-action=b']
Pip subprocess output:
Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cu113
Could not fetch URL https://download.pytorch.org/whl/cu113/charset-normalizer/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='download.pytorch.org', port=443): Max retries exceeded with url: /whl/cu113/charset-normalizer/ (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self signed certificate in certificate chain (_ssl.c:1129)'))) - skipping
```

This is because you cannot access the link and download it via `wget` or `curl`. Rather than downloading it automaticlly, you should get each whl-files (herein, torch, torchvision, and so on) directly from the link (e.g. https://download.pytorch.org/whl/cu113).

After downloading them, activate your conda environment and install them using `pip`. Here is the example: 

```bash
pip install torchvision-0.13.1+cu113-cp39-cp39-win_amd64.whl
```

[CUDA in WIKI - GPUs supported](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)  
[VERSIONS OF PYTORCH](https://pytorch.org/get-started/previous-versions/)  
[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)  
[cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)  
[Examples of GPytorch](https://github.com/cornellius-gp/gpytorch/tree/master/examples)


### So far, I just briefly introduce the installation for the pre-requisite sofrwares. Currently, the main codes are developing and i'm supposed to post any trouble-shooting when i face with some problems including some errors of package installation. I would like you to wait until finishing this work. Since i developed them by myself without any other co-workers, the development would be taken for a while.
