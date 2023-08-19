# differential-privacy-for-runtime-data




# README

This README provides instructions for building and running a Docker container based on the provided Dockerfile.

## Prerequisites
To build and run the Docker container, you need to have Docker installed on your system. You can download and install Docker from the official Docker website: [https://www.docker.com/](https://www.docker.com/).

## Building the Docker Image
To build the Docker image, follow these steps:

1. Clone or download the repository containing the Dockerfile and the application code.
2. Open a terminal or command prompt and navigate to the root directory of the cloned/downloaded repository.
3. Run the following command to build the Docker image: 

```docker build -t app .```

## Running the Docker Container
After successfully building the Docker image, you can run the Docker container with the following steps:

1. Open a terminal or command prompt.
2. Run the following command to start the Docker container:

```docker run -p 8050:8050 app```


_Note: [CUDA](https://developer.nvidia.com/how-to-cuda-python) unavailable when using Docker. This may lead to significantly longer training times._



## Alternatively, run in venv

1. Install venv: ```python -m venv venv```

2. Activate venv

    2.a. Windows: ```venv\Scripts\activate```

    2.b. Mac: ```source venv/bin/activate```

3. Install requirements: ```pip install -r requirements.txt```

4. Navigate to dashboard folder: ```cd dashboard```

5. Run app.py: ```python app.py```
s
Live demo: https://zunzer.github.io 
