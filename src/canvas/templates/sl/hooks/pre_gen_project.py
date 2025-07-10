import subprocess
import os
import sys

if "{{ cookiecutter.documentation }}" == "mkdocs":
    command = [sys.executable, '-m', 'mkdocs', 'new', '.', '--quiet']
    subprocess.run(command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
elif "{{ cookiecutter.documentation }}" == "sphinx":
    os.mkdir('docs/')
    command = [
        sys.executable,
        '-m',
        'sphinx.cmd.quickstart',
        '-q',
        '-p', "{{ cookiecutter.project_name }}",
        '-a', "{{ cookiecutter.author_name }}",
        '--no-makefile',
        '--no-batchfile',
        'docs/'
    ]
    subprocess.run(command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
