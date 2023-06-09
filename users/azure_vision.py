from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

# Replace with your key and endpoint
key = "<your-key>"
endpoint = "<your-endpoint>"

# Instantiate a client
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
