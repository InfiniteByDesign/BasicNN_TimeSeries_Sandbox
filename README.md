# BasicNN_TimeSeries_Sandbox
Sandbox to test different NN architectures on Time Series data


-------------------------------------
1.  How to use
-------------------------------------
All of the configuration options are available in the configuration.py script.  You can control the NN size, learning rate, etc.

Three example datasets are provided and no customization is needed if using one of them.  If you want to use your own dataset then you will need to modify the CSVReady.py according to your dataset properties.

If you want to modify the activation layer of the NN, then open the MLP_Definition.py file.

If you want to modify the cost function, optimization function, or the charts at the end, the modify the Main_NN.py and Main_NARX.py files.

-------------------------------------
2.  Executable Scripts
-------------------------------------
Main_NN.py

  A basic multilayer perceptron NN

Main_NARX.py

  A NARX mlp NN

-------------------------------------
3.  Data Set
-------------------------------------
The provided dataset have multiple samples for the inputs and multiple samples for the outputs.  The sample files also contain versions of the input samples with missing data and a set frequency.

-------------------------------------
4.  Data Import Script
-------------------------------------
CSVReader.py is customized to import your specific dataset.  This script reads the data and stores the columns into the appropriate x and y arrays.
-------------------------------------
5.  Helper function script
-------------------------------------
This script contains functions to manipulate the data for consumption by the NN.
