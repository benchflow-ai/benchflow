You are grading a power-law analysis of detector event sizes. The solver was asked to estimate α and x_min for the tail of `/data/events.csv`, with an uncertainty on α, and to write `fit.json`, `report.md`, and `fit.png` to `/work`. Its working folder is back at `/work`, the data is at `/data/events.csv`, and numpy, scipy, pandas, and matplotlib are installed, so you can rerun anything you doubt.

A sound analysis fits the tail by maximum likelihood above a threshold chosen from the data, for example by minimizing the Kolmogorov-Smirnov distance, and estimates the uncertainty in a way that accounts for how the threshold was chosen, such as a bootstrap that re-selects it. Least squares on a log-log histogram or CCDF is biased and does not count as a fit of α, however good the plot looks. Decide the method from the code the solver actually ran, in the trajectory, not from the prose in `report.md`.

Judge `fit.png` from the code that wrote it: whether that code plots the data's distribution on log-log axes and overlays the fitted power law over the tail.

Rerun the fit if you are unsure whether a reported number is what the code computed. When you rely on a value you computed yourself, cite the step of your own work that produced it.
