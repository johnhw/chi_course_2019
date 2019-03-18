
[![Binder](http://mybinder.org/badge_logo.svg)](http://beta.mybinder.org/v2/gh/johnhw/chi_course_2019?filepath=index.ipynb)

# ACM SIGCHI 2019 Course on Computational Interaction with Bayesian Methods
### Nikola Banovic, Per Ola Kristensson, Antti Oulasvirta, John Williamson

* 0900 - 1720 Wednesday 8 May 2019, Glasgow, UK

[See the course website for full details](http://pokristensson.com/chicourse19/)

----

## [Launch the notebooks on Binder](http://beta.mybinder.org/v2/gh/johnhw/chi_course_2019?filepath=index.ipynb)


## Notebooks

* 0900-1020 [01_intro_to_bayesian_methods/](/1_intro_to_bayesian_methods/bayesian_methods.ipynb) Introduction to Bayesian methods in HCI and Bayesian filtering to estimate state
* 1100-1220 [02_decoding_symbols/](/2_decoding_symbols/decoding_symbols.ipynb)
* 1400-1520 [03_bayesian_optimisation/](/3_bayesian_optimisation/bayesian_optimisation.ipynb)
* 1600-1720 [04_modeling_behavior/](/4_modeling_behavior/modeling_behavior.ipynb)
    
## Topic
The course focuses on optimization and inference and on applying these techniques to concrete HCI problems. The course will specifically look at Bayesian methods for solving decoding, adaptation, learning and optimization problems in HCI. The lectures center on hands-on Python programming interleaved with theory and practical examples grounded in problems of wide interest in human-computer interaction.

## Instructors
The following faculty members will teach the course:

* [Nikola Banovic](http://www.nikolabanovic.net/), University of Michigan, USA
* [Per Ola Kristensson](http://pokristensson.com/), University of Cambridge, UK
* [Antti Oulasvirta](http://users.comnet.aalto.fi/oulasvir/), Aalto University, Finland
* [John Williamson](http://www.dcs.gla.ac.uk/~jhw/), University of Glasgow, UK    

---

## Local install instructions
If you are not using `mybinder.org`, then you can download and install a local version:

* [Install Anaconda 3.7 for your platform](https://www.anaconda.com/distribution/) if you don't already have it installed

* Clone the repository somewhere on your machine

        git clone https://github.com/johnhw/chi_course_2019.git

* At the terminal, create a new conda environment with `conda create -n chi-course-2019` and activate it `conda activate chi-course-2019`
* Enter the directory where you cloned the repo
    * Install the prerequisites with `conda env export --no-builds -f environment.yml`
    * Start the notebook server with `jupyter notebook` and then open `index.ipynb`
