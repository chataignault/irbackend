from enum import Enum

from ...utils import *

class DataFolder(str, Enum):
    RAW = "raw_data"
    INGESTED = "ingested_data"
