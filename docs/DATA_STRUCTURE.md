# Dataset Layout

Full datasets are not included in this repository.

The original experiments used:

- Oxford RobotCar
- Nordland
- Pittsburgh30k

Set your data root in scripts, for example:

```text
DATA/
├─ robotcar_place/
├─ nordland_place/
└─ pitts30k_place/
```

The actual loaders are under:

```text
src/datasets/lreid_dataset/datasets/
```

Please adapt paths according to your local data organization.
