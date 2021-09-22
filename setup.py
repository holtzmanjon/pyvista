from setuptools import setup

setup(
    name='astro-pyvista',
    version='0.1.0',    
    description='Astronomical image processing',
    url='https://github.com/holtzmanjon/pyvista',
    author='Jon Holtzman',
    author_email='holtz@nmsu.edu',
    license='BSD 2-clause',
    packages=['pyvista','tools'],
    package_dir={"pyvista": "python/pyvista", "tools" : "python/tools/python/tools"},
    install_requires=[
                      'numpy',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)

