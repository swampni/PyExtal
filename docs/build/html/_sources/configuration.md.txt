# Input `.dat` File Configuration

The `.dat` file is the primary input for the Bloch wave simulation, defining the crystal structure, experimental conditions, and calculation parameters. The file is parsed sequentially, and keywords must appear in the correct order. Lines starting with `#` are treated as comments.

## 1. Crystal Structure Definition

This section defines the crystallographic properties of the sample.

### Title Line
The first line of the file is a descriptive title.
```
#Cu2O
```

### Crystal, Cell, Atom, and Space Group
These keywords define the fundamental crystal structure. This block is parsed by an internal `xtal1` routine.

- `crystal [name] : [options]`
  - Defines the crystal name. A colon `:` can be used to separate the name from optional keywords.
  - **Options**:
    - `dw = [type]`: Specifies the format for the Debye-Waller (temperature) factors provided on the `atom` lines. If omitted, `iso` is the default.
      - `iso`: Isotropic B-factor ($B_{iso}$). The `atom` line expects one value for the DW factor.
      - `bij`: Anisotropic beta factors ($\beta_{ij}$). The `atom` line expects 6 values ($\beta_{11}, \beta_{22}, \beta_{33}, \beta_{12}, \beta_{13}, \beta_{23}$).
      - `uij`: Anisotropic U factors ($U_{ij}$). The `atom` line expects 6 values ($U_{11}, U_{22}, U_{33}, U_{12}, U_{13}, U_{23}$).
    - `occ = [type]`: Specifies how atomic site occupancy is handled. If omitted, `1` (full occupancy) is the default.
      - `1`: Assumes full occupancy (value is 1.0). The occupancy value is not read from the `atom` line.
      - `par`: Partial occupancy. The occupancy value must be provided as the last number on each `atom` line.

- `cell [a] [b] [c] [alpha] [beta] [gamma]`
  - Unit cell parameters: lattice constants in Angstroms, angles in degrees.

- `atom [label] [x] [y] [z] [dw_factor(s)] [occupancy]`
  - Defines an atom in the unit cell.
  - `label`: Chemical symbol (e.g., `Cu`, `O`).
  - `x, y, z`: Fractional coordinates.
  - `dw_factor(s)`: One or six values for the Debye-Waller factor, depending on the `dw` setting.
  - `occupancy`: (Optional) The site occupancy, required only if `occ = par` is set.

- `spg [number] [setting]`
  - Space group number and origin setting.

**Example:**
```
# The following line sets isotropic DW factors and full occupancy by default.
crystal Cu2O : dw = iso occ = 1

cell 4.259600 4.259600 4.259600 90.000000 90.000000 90.000000

# The atom lines provide one value for the DW factor (B_iso)
atom Cu  0.00 0.00 0.00 0.6813
atom O   0.25 0.25 0.25 0.7757
spg 224 2
```

## 2. Experimental & Diffraction Parameters

This section sets up the electron beam and diffraction geometry.

- `v0 [potential]` (Optional)
    - Mean inner potential of the crystal in Volts.
- `hv [voltage]` (**Mandatory**)
    - The accelerating voltage of the electron microscope in kV.
- `zone [h] [k] [l]` (**Mandatory**)
    - The integer Miller indices of the zone axis pointing towards the electron source.
- `norm [h] [k] [l]` (**Mandatory**)
    - The integer Miller indices of the sample surface normal, pointing out of the exit surface.
- `kt [kx] [ky] [kz]` (**Mandatory**)
    - The tangential component of the incident wavevector, defining the center of the CBED pattern.
- `xaxis [h] [k] [l]` (**Mandatory**)
    - A reciprocal lattice vector that will be oriented horizontally in the output diffraction pattern.
- `conv [angle]` (**Mandatory**)
    - The convergence semi-angle in unit of xaxis. For example, if conv 0.5, the (xaxis) disk will be touching the bright field disk.
- `samp [pixels]` (**Mandatory**)
    - The number of pixels to sample from the center to the edge of the convergent beam disk.

**Example:**
```
hv 119.523003
zone 0 1 2
norm 0 1 2
kt 2.59327 -8.68843 4.34422
xaxis 1 0 0
conv 2
samp 60
```

## 3. Beam Selection

This section determines which diffracted beams (reflections) are included in the Bloch wave calculation matrix. You must use **one** of the following two methods.

### Method 1: Manual Beam Selection (`hkl`)
Manually specify every beam to be included.

- `hkl [nbeams]`
  - Keyword followed by the total number of beams.
- The following `[nbeams]` lines must contain the `h k l` indices for each beam.

**Example:**
```
hkl 312
0   0   0
-2   0   0
2   0   0
... (309 more lines) ...
```

### Method 2: Automatic Beam Selection (`sele`)
Select beams automatically based on geometric criteria.
Parameters for automatic beam selection (described in [Zuo and Weickenmeier, 1990](https://www.sciencedirect.com/science/article/abs/pii/030439919400190X)):

- `sele [sgmax] [gmax] [bmin] [kflag]`
  - `sgmax`: Maximum excitation error to include a beam.
  - `gmax`: Maximum length of the g-vector to include.
  - `bmin`: Bethe potential cutoff.
  - `kflag`: keep 0.

## 4. Calculation & Output Parameters

These keywords control the accuracy of the calculation and define which beams are saved in the output.

- `sgmin [value]`
  - **Mandatory**. Minimum excitation error. Beams with excitation errors smaller than `sgmin` are included in the full diagonalization. beams are included if $2KS_g < S_{g_{min}}$
- `omeg [value]`
  - **Mandatory**. A parameter related to the Bethe potential that affects accuracy. Larger is more accurate. beams are included if $\Omega_g < \Omega$
- `aper [radius]` or `out [nbeams]`
    - **Mandatory**. Choose one method to specify output beams:
        - `aper [radius]`: Simulates an objective aperture, including all beams within this `radius` (in Å⁻¹).
        - `out [nbeams]`: Manually specify `[nbeams]` beams. Must be followed by `[nbeams]` lines containing the `h k l` indices.
- `abs [potential]`
  - **Mandatory**. Mean absorption potential in Volts.

**Example from `Cu2O_LARBED.dat`:**
```
sgmax 3.5
sgmin 0.15
omeg 25

aper 1.0
abs 0.0
```

## 5. Structure Factor Adjustment (Optional)

This section allows for manual adjustment of structure factors for refinement purposes.

- `adj [nref]`
  - Keyword followed by the number of reflections whose structure factors will be adjusted.
- The following `[nref]` lines contain the reflection and its new structure factor values:
  - `h k l U_g_amp U_g_phase U'_g_amp U'_g_phase`  
phase is in degree


