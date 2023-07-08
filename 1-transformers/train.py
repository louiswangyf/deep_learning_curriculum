import os
import torch
import numpy as np
from config import GPTConfig
config = GPTConfig()
from gpt import GPT
import time

class Trainer:

    @staticmethod
    def get_default_config():
        config = GPTConfig()
        return config
    
    