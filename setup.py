import json
import os
import setuptools
import shutil

from pathlib import Path
from setuptools.command.install import install

# Load requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        try:
             # cwd is temporary folder
            usr_home = Path.home()
            cognival_usr_dir = usr_home / '.cognival'
            if os.path.exists(cognival_usr_dir):
                raise FileExistsError
            base = Path('user')
            shutil.copy('README.md', (base / 'README.md')),
            for path, file_ in [('.', 'README.md'),
                                ('embeddings', 'README.md'),
                                ('cognitive_sources', 'README.md'),
                                ('results', 'README.md'),
                                ('configurations', 'cognitive_sources.json'),
                                ('configurations', 'demo_config.json'),
                                ('configurations', 'embedding_registry.json'),
                                ('configurations', 'reference_config.json'),
                                ('configurations', 'standard_vocab.txt'),
                                ('configurations', 'standard_sentences.txt'),
                                ('configurations', 'standard_sent_vocab.txt')
                        ]:
                try:
                    os.makedirs(cognival_usr_dir / path)
                except FileExistsError:
                    pass
                shutil.copy((base / path / file_),
                        (cognival_usr_dir / path / file_))
        except FileExistsError:
            pass
        install.run(self)

setuptools.setup(
    name="cognival",
    version="0.2.2",
    author="Multiple authors",
    author_email="",
    description="CogniVal cognitive embedding evaluation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=['cognival/cognival'],
    install_requires=requirements,
    python_requires='>=3.7',
    include_package_data=True,
    cmdclass={
        'install':PostInstallCommand
    }
)
