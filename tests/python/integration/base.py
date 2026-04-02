import subprocess
from unittest import TestCase

class IntegrationTestCase(TestCase):

    def run_command(self, command):
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        return proc