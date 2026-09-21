# Dataset setup

The full audit uses **PIE** and **JAAD**. Raw images and videos are not included in this repository.

## PIE

Official resources:

- Repository and annotations: https://github.com/aras62/PIE
- Dataset page and videos: https://data.nvision2.eecs.yorku.ca/PIE_dataset/

Expected project layout:

```text
data/PIE/
├── annotations/
├── PIE_clips/
│   ├── set01/
│   ├── set02/
│   └── ...
└── images/                  # optional extracted-frame cache
    ├── set01/
    │   └── video_XXXX/
    └── ...
```

The loader first checks for extracted PNG frames under `images/`; otherwise it seeks directly into the corresponding MP4 file under `PIE_clips/`.

The official PIE Python interface (`pie_data.py`, distributed with the PIE repository) must be importable by Python.

## JAAD

Official resources:

- Repository and annotations: https://github.com/ykotseruba/JAAD
- Dataset page and videos: https://data.nvision2.eecs.yorku.ca/JAAD_dataset/

Expected project layout:

```text
data/JAAD/
├── annotations/
├── JAAD_clips/
│   ├── video_0001.mp4
│   └── ...
└── images/                  # optional extracted-frame cache
    ├── video_0001/
    └── ...
```

The official JAAD Python interface (`jaad_data.py`) must be importable by Python.

## Making the interfaces importable

The simplest options are either:

1. copy `pie_data.py` and `jaad_data.py` from the official repositories into `src/`; or
2. keep the official repositories separate and add the folders containing those modules to `PYTHONPATH`.

No modifications to the official annotations are required by the audit code.

## Licensing

PIE and JAAD remain third-party datasets. Follow their original license and citation requirements. See the root-level `THIRD_PARTY_NOTICES.md`.
