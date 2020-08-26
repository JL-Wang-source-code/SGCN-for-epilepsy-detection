# SGCN-for-epilepsy-detection
This repository holds the source code of SGCN model.
The code consists of two parts: Frequency-domain_Complex_network and SGCN.

Dataset

Due to the size limitation of upload file, we provide website of the Bonn dataset：http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3&changelang=3.

Frequency-domain_Complex_network

This part contains two code files, LVG.m and LPHVG.m. In LPVG and LPHVG, we set the variable L of limited number of penetrations. When you set L to 0, you can build VG and HVG.

SGCN

This part contains two code files, Node_aggregation.m and SGCN.py. The main file is SGCN.py. To use this classifier, please generate the complex networks in advance, and make sure that you have pytorch, Python 3 and all the packages we have used installed.

Next, please take the following two steps.

  Step 1. Change the path in line 77 of SGCN.py to the path of input data.
  
  Step 2. Run the command in your command council.
