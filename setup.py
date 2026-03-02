from setuptools import setup

setup(
    name='geoxts',
    version='0.1.0',    
    description='A benchmark library for time series classification with gradcam visualisations, focused on geological data.',
    url='https://gitlab.lisn.upsaclay.fr/salimath/geox',
    author='Shwetha Salimath',
    author_email='shwetha.salimath@lisn.fr',
    license='CC BY-NC-SA',
    packages=['geoxts'],
    install_requires=['pandas',
                      'numpy', 
                      'fastparquet',
                      'tsai',
                      'fastai',
                      'statistics',
                      'pathlib',
                      'matplotlib',
                      'torch',
                      'scikit-learn',
                      'dtaidistance',
                      'threadpoolctl'                    
                      ],

    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: CC BY-NC-SA', 
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)