# Singum Traffic Flow Research

Manhattan traffic flow prediction using the Singum model. This codebase generates Figures 5c and 5d for the PNAS publication "Homogenization of a two-dimensional network for flow prediction and network routing."

## Paper Equations → Code Mapping

| Paper Equation | Description | Code Function | Lines |
|----------------|-------------|---------------|-------|
| **Eq. 2** | Constitutive relation: `q_i = -k_ij V_,j` | Implicit in FEM formulation | — |
| **Eq. 11** | Governing PDE: `k_ij V_,ij = 0` | `solve_fem_anisotropic()` | 273–309 |
| **Eq. 20** | CB rule: `k¹_ij = Σ KᴵA x_i x_j / (Vs l_p)` | `build_K_from_voronoi()` | 1230–1244 |
| **Eq. 21** | Non-symmetric correction: `k²_ij` | `build_K_from_voronoi()` | 1246–1271 |

### FEM Solver Details

The anisotropic diffusion PDE ∇·(K∇V) = 0 is solved using P1 finite elements:

| Function | Purpose | Lines |
|----------|---------|-------|
| `_create_fem_mesh()` | Delaunay triangulation (~163k nodes, ~327k triangles) | 148–183 |
| `_element_stiffness_anisotropic()` | P1 element stiffness matrix with K tensor | 186–228 |
| `_assemble_fem_system()` | Global stiffness matrix assembly | 231–270 |
| `solve_fem_anisotropic()` | Main FEM solver with Dirichlet BCs | 273–309 |

### K Tensor Computation

The effective conductivity `k_ij = k¹_ij + k²_ij` is computed in `build_K_from_voronoi()`:

- **k¹ (Cauchy-Born rule)**: Standard homogenization assuming uniform gradient
- **k² (Correction term)**: Accounts for non-central-symmetric networks like Manhattan's irregular road grid

## Directory Structure

```
singum_research/
├── case2.py                  # Main pipeline (run this)
├── singum.py                 # Voronoi cell generation
├── load_data.py              # Data loading utilities
├── symmetric_transportation.py
│
├── data/                     # Input data (required)
│   ├── filtered_roads.geojson
│   ├── manhattan_boundary.geojson
│   ├── nyc_boundary.geojson
│   ├── singum_voronoi.geojson
│   └── lion/                 # LION geodatabase
│
├── output/                   # Generated files
│   ├── fig5c_nba_*.png       # Figure 5c outputs
│   ├── fig5d_gw_commute.png  # Figure 5d output
│   └── ...                   # Intermediate files
│
├── docs/                     # Documentation
│   ├── case2_functions_documentation.pdf
│   └── codebase_organization_guide.pdf
│
└── archive/                  # Old reference files
```

## Prerequisites

### Python Dependencies

Install required packages:

```bash
pip install numpy pandas geopandas matplotlib scipy shapely fiona rasterstats
```

Or create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas geopandas matplotlib scipy shapely fiona rasterstats
```

### Required Data Files

Ensure these files exist in `data/`:

| File | Description | Size |
|------|-------------|------|
| `filtered_roads.geojson` | NYC road network | ~55 MB |
| `manhattan_boundary.geojson` | Manhattan polygon | ~170 KB |
| `singum_voronoi.geojson` | Voronoi cells | ~8 MB |
| `lion/` | LION geodatabase | (directory) |

## Quick Start

### Run Everything

```bash
python case2.py
```

This runs the full pipeline:
1. Preprocesses road data (clips to Manhattan)
2. Builds the road network graph
3. Computes K tensors with k² correction (Eqs. 20–21)
4. Solves continuum PDE using FEM (Eq. 11)
5. Generates all figures (5c and 5d)

### Run Individual Figures

In Python:

```python
from case2 import run_fig5c, run_fig5c_arrival, run_fig5d

# Figure 5c: NBA game dispersal (MSG -> destinations)
run_fig5c()

# Figure 5c reverse: Arrival at MSG (sources -> MSG)
run_fig5c_arrival()

# Figure 5d: GW Bridge to Columbia commute
run_fig5d()
```

Or from command line:

```bash
# Run just Figure 5c
python -c "from case2 import run_fig5c; run_fig5c()"

# Run just Figure 5d
python -c "from case2 import run_fig5d; run_fig5d()"
```

## Output Files

After running, `output/` will contain:

### Figures

| File | Description |
|------|-------------|
| `fig5c_nba_grand_central.png` | MSG → Grand Central |
| `fig5c_nba_columbia.png` | MSG → Columbia |
| `fig5c_nba_nyu.png` | MSG → NYU |
| `fig5c_nba_times_square.png` | MSG → Times Square |
| `fig5c_nba_wall_street.png` | MSG → Wall Street |
| `fig5c_nba_arrival_*.png` | Reverse routes (→ MSG) |
| `fig5c_nba_arrival_combined.png` | All arrival routes combined |
| `fig5d_gw_commute.png` | GW Bridge → Columbia |

### Intermediate Files

| File | Description |
|------|-------------|
| `graph_edges.txt` | Road network graph |
| `nodes_points_2263.geojson` | Node coordinates |
| `nodes_with_phi_2263.geojson` | Nodes with potential values |
| `singum_voronoi_with_K.geojson` | Voronoi with K tensors |
| `K_centroids_2263.csv` | Conductivity tensor data |

## Regenerating Voronoi Cells

If `data/singum_voronoi.geojson` doesn't exist or needs to be regenerated:

```bash
python singum.py
```

This reads the LION geodatabase and creates Voronoi cells for each road network node.

## Configuration

Key parameters in `case2.py`:

```python
# Landmark locations (lon, lat)
MSG_LONLAT = (-73.9934, 40.7505)           # Madison Square Garden
COLUMBIA_LONLAT = (-73.9626, 40.8075)      # Columbia University
GW_BRIDGE_LONLAT = (-73.9527, 40.8517)     # George Washington Bridge

# Scenario parameters
MSG_RADIUS_FT = 400.0      # Source/sink region radius
GW_COMMUTERS = 5000        # Number of commuters (Fig 5d)
NBA_ATTENDEES = 20000      # Number of attendees (Fig 5c)

# Grid resolution for continuum PDE
GRID_NX, GRID_NY = 700, 1050
```

## Troubleshooting

### "No module named 'geopandas'"

Install dependencies:
```bash
pip install geopandas
```

### "FileNotFoundError: data/filtered_roads.geojson"

Ensure all required data files are in the `data/` directory.

### Slow performance

The FEM solver (P1 triangular elements, ~163k nodes) can take 1-2 minutes. This is normal for the 700×1050 grid.

### Memory issues

Reduce grid resolution in `case2.py`:
```python
GRID_NX, GRID_NY = 350, 525  # Half resolution
```

## Documentation

See `docs/` for detailed documentation:

- `case2_functions_documentation.pdf` - Function reference and dependency graph
- `codebase_organization_guide.pdf` - Codebase structure and file descriptions

## Citation

If you use this code, please cite the PNAS paper:

```
Shen, A., Liu, C., Wu, K., Lin, B., John, S., Xu, A., Kim, B.-W., & Yin, H. (2024).
Homogenization of a two-dimensional network for flow prediction and network routing.
Proceedings of the National Academy of Sciences.
```
