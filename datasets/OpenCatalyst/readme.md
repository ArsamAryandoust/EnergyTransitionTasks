# Open Catalyst


n := number of data points <br />
a := number of atoms <br />

| Variable | Description | Format |
| --- | ----------- | ----------- |
| x_s | Space-variant features: the atomic number of every atom in the given adsorbate-catalyst system. Can be mapped to x_s_1, x_s_2 and x_s_3 from additional data. | (n, a) |
| x_st | Space-time-variant features: the 3D position of every atom in the given adsorbate-catalyst system. | (n, 3a) |
| y_t | Time-variant labels: resulting energy (-s2ef) or relaxed energy (-is2re) of the given adsorbate-catalyst system. This label is only available for -s2ef and -is2re tasks. | (n, 1) |
| y_st | Space-time-variant labels: 3D position of relaxed structure (-is2rs) or 3D position of forces for given adsorbate-catalyst system (-s2ef). This label is only available for -is2rs and -s2ef tasks. | (n, 3a) |


| Additional data | Description | Format |
| --- | ----------- | ----------- |
| x_s_1 | Space-variant features 1: 8 numeric properties of each atom consisting of atomic mass, electronegativity, atomic radius, ionization energy, electron affinity, melting point, boiling point and density| (8, a) |
| x_s_2 | Space-variant features 2: ordinally encoded information about the standard state and group block of an atom| (2, a) |
| x_s_3 | Space-variant features 3: one-hot encoded information about oxidation states of an atom | (2, a) |



