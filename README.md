# Lattice-based Grain Segmentation for HRTEM images and videos

<a name="readme-top"></a>

<br />
<div align="center">
  <h3 align="center">an automated atomic resolution TEM image process and analysis pipeline</h3>
  <p align="center">
    <br />
    <br />
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Atomic resolution transmission electron microscopy (TEM) image gives tremendous structural imformation of the materials. However, only limited information was utilized in most of the TEM related researches due to the great challenge in quantatitive analysis of TEM images. And the quality of data analysis relies very much on the experience of the researchers. For in situ atomic resolution TEM, the challange is even greater since the dataset typically contains thousands of images.  
To tackle this issue, automating image process and data mining process is necessary. One of the most important tasks is the grain segmentation, which could give the morphologies of nanocrystals and the grain boundaries of polycrystals. This is also the prerequsite step for automated strain analysis, defect analysis, etc. 
Therefore, for this project, we are motivated to develop automated grain segmentation alogrithm for atomic resolution TEM images (HRTEM, STEM, etc.) of crystalline samples based on lattice pattern. 
There are mainly two modules: atom orientation mapping and grain segmentation. 

### Atom orientation mapping

In HRTEM images, atom columns are visible when the crystal is on zone axis. The orientation of atoms can be determined from the arrangement of the neighbor atoms. Here we borrow the idea of template matching (P. M. Larsen et al. Robust Structural Identification via Polyhedral Template Matching, Modelling Simul. Mater. Sci. Eng. 24, 055007 (2016), doi:10.1088/0965-0393/24/5/055007). Following is the flowchart:  
<img src="image/OrientationMapping.png" alt="Logo" width="500" height="500">  
Image is first processed by FFT filter. Then atoms in the image are detected by LoG blob detection. Voronoi analysis is used to represent the local environment of atoms. Template matching is therefore conducted on each voronoi cell of the atoms, yielding the orientation of the atoms. In this way, we get the atom orientation mapping.  
For the usage, please refer to Atom_orientation_mapping.ipynb.

### Grain segmentation


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [x] Atom orientation mapping
- [x] Grain segmentation
- [ ] Polycrystal strain mapping
- [ ] Crystallographic defect detection


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Yuan Tian - tiany17@uci.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project is supported by Army Research Office (ARO) Project about grain boundary dynamics. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



