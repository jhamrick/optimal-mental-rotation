from termcolor import colored
import subprocess


def run_command(logger, cmd):
    logger.debug(colored("Running %s" % " ".join(cmd), 'blue'))
    code = subprocess.call(cmd)
    if code != 0:
        raise RuntimeError("Process exited abnormally: %d" % code)
