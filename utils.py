import sys
import os


class DualLogger(object):
    """
    Redirects stdout to both the terminal and a log file.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logger(output_dir, base_name):
    """
    Sets up the logger to write to a file in output_dir.
    Returns the full path to the log file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file_path = os.path.join(output_dir, f"log_{base_name}.txt")
    sys.stdout = DualLogger(log_file_path)
    return log_file_path
