# Hubble data

The hubble data indicate the distance between Earth and a galaxy, and the speed of each galaxy moving away from Earth.<br>
Here we're interested in the relation between the distance and the recession speed of galaxies.<br>
This relationship was first discovered by Erwin Hubble (https://en.wikipedia.org/wiki/Edwin_Hubble) in 1929. See the article https://www.pnas.org/content/15/3/168.

In this code, the relation between distance and speed is determined by a linear interpolation, which is first explicity computed by solving the normal equation - providing the exact solution. Then the linear regression of Scikit-learn Python module is adopted, and results are compared with the exact solution.


