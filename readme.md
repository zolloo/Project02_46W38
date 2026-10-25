Project 02 – Turbie Dynamic Simulation

1. Overview

This project simulates the blade and tower fore-aft responses of a simplified 2-DOF wind turbine model under different wind speeds and turbulence intensities (TI).

The model is implemented in turbie_mod.py and simulated using scipy.integrate.solve_ivp in main.py.

2. How to Run

Run main.py with the supplied data in the inputs folder:

python main.py

Simulation results, statistics and plots are saved in the outputs folder.

3. Results and Discussion

Effect of wind speed

Mean blade and tower displacements increase with wind speed up to around 11 m/s, then decrease at higher wind speeds. This is mainly due to the reduction in thrust coefficient (CT), which leads to lower aerodynamic thrust despite increasing wind speed.

Effect of turbulence intensity

Higher TI leads to larger standard deviations of both blade and tower displacements, as greater wind fluctuations produce stronger variations in aerodynamic loading. Mean displacements are much less affected by TI.

Blade and tower responses

Blade displacements are generally larger than tower displacements. The aerodynamic thrust acts directly on the blades, while the tower responds through the coupled stiffness and damping of the system.

Overall, mean displacement is mainly influenced by aerodynamic thrust, while displacement fluctuations are strongly affected by turbulence intensity and structural dynamics.