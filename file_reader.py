"""Module for reading files, e.g. data csv files."""
import numpy as np

def readfile(name: str) -> tuple[np.ndarray]:
    """Read a csv file, return two arrays: time and intensity."""
    time = []
    intensity = []
    with open(name, "r", encoding="utf-8") as csv_file:
        for line in csv_file.readlines():
            line = line.split(',')
            time.append(float(line[0]))
            intensity.append(-float(line[1]))
    return np.array(time),np.array(intensity)
