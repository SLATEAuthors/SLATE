from setuptools import setup

setup(
    name='sumie',
    packages=['sumie'], 
    package_dir={'sumie': 'sumie'}, 
    version='0.1.0',
    description='Package for SLATE: A Sequence Labeling Approach for Task Extraction from Free-form Content',
    package_data = {'sumie' : ['models/data/*']},
    include_package_data=True, 
)
