from setuptools import setup

setup(name='data_tools',
      version='0.1',
      description='This is a collection of tools that I find useful '\
        'and time-saving in tasks like directory structure creation. '\
        'Packaging this up may be overkill, but I wanted to preserve '\
        'at least some functionality for future projects.\n'\
        'It currently lives in the repository for a related project.'
      #long_description=readme(),
      classifiers=[
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Astronomy',
        #This may need to be changed...
      ],
      #keywords='data processing directory structure'
      #url='https://github.com/jacksonlanecole/WallinCode/'\
      #'tree/master/data_tools',
      author='Jackson Cole',
      author_email='research@jacksoncole.io',
      #packages=['data'],
      #install_requires=[
      #    'markdown',
      #],
      include_package_data=True,
      zip_safe=False)
