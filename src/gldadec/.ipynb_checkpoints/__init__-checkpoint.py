# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 05:51:00 2020

@author: I.Azuma
"""
from setuptools import setup  
from Cython.Build import cythonize  
 
setup(  
    ext_modules = cythonize("_lda_basic.pyx")  
)
