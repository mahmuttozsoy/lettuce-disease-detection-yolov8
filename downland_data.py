
from roboflow import Roboflow
import os

rf = Roboflow(api_key=os.getenv("kqNNYAsvjCAdGi4Ei7FG"))
project = rf.workspace("spice-team-2g-extra").project("lettuce-objdet-x3")
version = project.version(2)
dataset = version.download("yolov8")
print("Dataset downloaded successfully.")
