import cv2
from ML_utils import *
from grid import *
import time

# Path in which the trained model is saved
TRAINED_MODEL_PATH = 'Trained Models/model.ckpt'

def display(string):
  print string
