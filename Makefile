# Define variables
PYTHON = python
PIP = pip
DOCKER = docker
FLASK_APP = app.py
MODEL_SCRIPT = train_model.py
DOCKER_IMAGE_NAME = my-flask-app
PORT = 4000:5000

# Define the default make target
all: setup train docker-build docker-run

# Setup the Python virtual environment and install dependencies
setup:
	@echo "Setting up the virtual environment and installing requirements..."
	$(PYTHON) -m venv venv
	./venv/bin/activate
	$(PIP) install -r requirements.txt

# Train the model
train:
	@echo "Training the model..."
	$(PYTHON) $(MODEL_SCRIPT)

# Build the Docker image
docker-build:
	@echo "Building the Docker image..."
	$(DOCKER) build -t $(DOCKER_IMAGE_NAME) .

# Run the Docker container
docker-run:
	@echo "Running the Docker container..."
	$(DOCKER) run -p $(PORT) $(DOCKER_IMAGE_NAME)

# Run the Flask application locally
run-flask:
	@echo "Running the Flask application locally..."
	FLASK_APP=$(FLASK_APP) FLASK_ENV=development $(PYTHON) -m flask run

.PHONY: all setup train docker-build docker-run run-flask
