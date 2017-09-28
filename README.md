# Toponym Matching with Deep Neural Networks

This work was originally developed by Rui Santos in the context of an MSc thesis project at Instituto Superior Técnico of the University of Lisbon. His MSc research concerned with the development of methods for automatically georeferencing historical itineraries given in a tabular format.

Further developments are currently being made by Alexandre Marinho, in the context of an MSc thesis related to string matching for duplicate detection.

Toponym matching refers to the problem of matching a pair of strings that refer to the same real-world location. Several approaches were tested in the context of toponym matching experiments involving data colected from GeoNames:

* Methods based on classical string similarity functions;
* Methods based on supervised machine learning for combining multiple similarity features;
* Methods based on supervised machine learning leveraging deep neural networks for encoding and matching strings.

The source code provided in this repository alows one to reproduce the results described on two publications concening with toponym matching:

   @article{Santos2017a,
      author = {Santos, Rui and Murrieta-Flores, Patricia and Martins, Bruno},
      journal = {International Journal of Digital Earth},
      title = {Combining Multiple String Similarity Metrics for Effective Toponym Matching},
      year = {2017}
   }
   
   @article{Santos2017b,
      author = {Santos, Rui and Murrieta-Flores, Patricia and Calado, P{\'{a}}vel and Martins, Bruno},
      journal = {International Journal of Geographic Information Systems},
      title = {Toponym Matching Through Deep Neural Networks},
      year = {2017}
   }

The 'dataset' folder contains the data used on the experiments that are reported on the publications above, consisting of 5 million pairs of toponyms collected from the GeoNames gazetteer.

To use of the machine learning models that combine string similarity features, refer to the *featureclassifiers.py*. The approaches based on deep neural networks are available in *deepneuralnetwork.py*. The code used to generate the dataset and to compute the string similarity measures is available in *datasetcreator.py*.

The source code was tested using Python 2.7, Keras 1.2.2, and Scikit-Learn 0.18.1.
