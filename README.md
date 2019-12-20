# Climate-Science-Analysis
Various climate science analysis and plots: Winds, Sea Ice, Empirical Orthogonal Functions, etc

Main code: CMIP5_Climate.py


* The climate model data are stored at UPenn's local server

Functions code: Behzadlib.py

* This code contains various analysis/plotting functions that are imported in the main code as needed


Final plotting products:
* Fig_Wind_Curl_GFDL-ESM2G.png = Curl of the wind, calculated as
Wind_Curl = ( D_Tau_Y / D_X ) - ( D_Tau_X / D_Y ) # D_X = (Lon_1 - Lon_2) * COS(Lat)

* Fig_Wind_Curl_f_WQuiver_GFDL-ESM2G.png = Ekman transport, equal to wind curl divided by coriolis parameter. The quivers are the wind direction - Wind_Crul / f , f = coriolis parameter = 2Wsin(LAT) , W = 7.292E-5 rad/s

* Fig_SeaIce_Arctic_monthly_GFDL-ESM2G.png = Arctic Sea Ice concentration average for each month - average of 1991-2000
