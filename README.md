**TrueBS**

This repository contains the code needed to reproduce the results of the research article called ``On Cost-effective, Reliable Coverage for LoS Communications in Urban Areas''
published on the Special Issue Recent Advances in the Design and Management of Reliable Communication Networks of the IEEE Journal Transactions on Network and Service Management.

**Dataset**

The dataset containing the morphological datasets (DSM and DTM) and the results is available on Zenodo (add link here).

**Database**

The database needed to store the vectorial dataset of the buildings can be recreated by installing the latest version of PostgresSQL + PostGis and by populating it with the OSM buildings data available here (link to geofabrik).

**How to replicate the results**
The optimal BS locations can be found in the related Zenodo repository. You can either analyze those or generate new ones by following the instruction in the next section.

To analyze the topologies you can execute the python script `BS_analysis.py`

**How to generate the optimal BS locations**

In order to generate the optimal BS locations, first, you need to install the python dependencies:

```
pip -r requirements.txt
```

Then, copy the `sim.yaml.example` file to `sim.yaml`, and adjust the content to your needs.

```
cp sim.yaml.exampe sim.yaml
```

Run the python script `TrueBS.py`, which will generate a set of optima BS locations in the `results/` folder.
