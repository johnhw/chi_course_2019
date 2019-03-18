
[![Binder](http://mybinder.org/badge_logo.svg)](http://beta.mybinder.org/v2/gh/johnhw/chi_course_2019?filepath=index.ipynb)

# ACM SIGCHI 2019 Course on Bayesian Methods on Interaction

[Launch the notebooks on Binder](http://beta.mybinder.org/v2/gh/johnhw/chi_course_2019?filepath=index.ipynb)

* 0900 - 1720 Wednesday 8 May 2019

## Notebooks

* [/intro_to_bayesian_methods](/intro_to_bayesian_methods/bayesian_methods.ipynb) Introduction to Bayesian methods in HCI and Bayesian filtering to estimate state
    

## Local install instructions
If you are not using `mybinder.org`, then you can download and install a local version:

* [Install Anaconda 3.7 for your platform](https://www.anaconda.com/distribution/) if you don't already have it installed

* Clone the repository somewhere on your machine

        git clone https://github.com/johnhw/chi_course_2019.git

* At the terminal, create a new conda environment with `conda create -n chi-course-2019` and activate it `conda activate chi-course-2019`
* Enter the directory where you cloned the repo
    * Install the prerequisites with `conda env export --no-builds -f environment.yml`
    * Start the notebook server with `jupyter notebook` and then open `index.ipynb`


## Topic
The course will focus on optimization and inference and on applying these techniques to concrete HCI problems. The course will specifically look at Bayesian methods for solving decoding, adaptation, learning and optimization problems in HCI. For more information, please see our course paper, which includes details of the materials taught and the relative timing.

## Instructors
The following faculty members will teach the course:

* [Nikola Banovic](http://www.nikolabanovic.net/), University of Michigan, USA
* [Per Ola Kristensson](http://pokristensson.com/), University of Cambridge, UK
* [Antti Oulasvirta](http://users.comnet.aalto.fi/oulasvir/), Aalto University, Finland
* [John Williamson](http://www.dcs.gla.ac.uk/~jhw/), University of Glasgow, UK

## Content

This course introduces computational methods in human–computer interaction. Computational interaction methods use computational thinking, abstraction, automation, and analysis, to explain and enhance interaction. This course introduces the theory and practice of computational interaction by teaching Bayesian methods for interaction across four wide areas of interest when designing computationally-driven user interfaces: decoding, adaptation, learning and optimization. The lectures center on hands-on Python programming interleaved with theory and practical examples grounded in problems of wide interest in human-computer interaction.


[See the course website for full details](http://pokristensson.com/chicourse19/)

## License
This material is licensed under the MIT license.