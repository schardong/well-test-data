# Well testing reservoir model learning
Data and code regarding the reservoir model inference from well test pressure derivative data.

## Requirements

Most of the code in this repository is written in Python 3. The experiments themselves are located in Jupyter Notebooks in the ```notebooks``` folder. The analytical scenario generation is written in Matlab/GNU Octave script and is located in the ```utils``` folder. The python requirements are listed in ```requirements.txt``` and can be installed using:

```bash
> pip install -r requirements.txt
```

The external requirements for this project are:

* Matlab or GNU Octave (for the analytical scenario generation)
* CMG IMEX and Results (for the numerical scenario generation)

These are only needed if you intend to generate more scenarios for further testing. If you want to run the experements, only the python requirements are needed.

## Scenario generation
This section describes the scenario generation processes used in our work, and how to run them.

### Analytical solutions
The analytical solutions are provided by the Matlab script ```utils/scenario_gen.m```. This script contians a series of functions to generate the scenarios accerding to different reservoir models. The main function is ```CALCULA_PWD``` which uses Stehfest algorithm to solve for the pressure values. The main loop of the algorithm is located at the end of the file, together with the algorithm's parameters. In order to generate scenarios for different models, change the function ```FT_PWD_(MODEL)_(EFFECT)``` called inside ```CALCULA_PWD```. At the end or the script, the data is saved in a comma-separated-values format.

In order to consolidane scenarios from several models in a single dataset, the notebook ```Analytical data test and merge.ipynb``` was created. It also contains some code to inspect the scenarios and manually check for correctness. For now, the notebook is fairly static, you would need to change the input dataset paths if you generated new scenarios with other models. This will be made more generic in a further version.

### Numerical solutions

## Reservoir model learning
