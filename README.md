# Cell migration CRISPRi screens in human neutrophils reveal regulators of context-dependent migration and differentiation state

[BioRxiv preprint, DOI: https://doi.org/10.1101/2022.12.16.520717](https://doi.org/10.1101/2022.12.16.520717)

## Branches
This repository contains two branches -- `master`, and
`publication`. The `master` branch contains additional files, including more
exploratory code and data that may not have made it into the final publication.
The final branch, `publication` (where you are now) contains
all data and code used to generate figures, clean and process data. This branch
should be sufficient to reproduce everything presented in this work.

## Repository Architecture
This repository is broken up into several directories and subdirectories. Please
see each directory for information regarding each file.

### `code`
All Python code used in this work is located here and is written to be
executed from its local directory. This directory contains two subdirectories
in which the code is organized.

1. **`processing`** | All code that was used in processing of raw datasets
   (genomics data, image data) and collation of screen datasets.
2. **`figures`** | All code used to generate the figures, both main text and supplemental.
    The scripts are written to be executed from this directory.

### `data`
This folder contains the processed data used throughout this work and used to generate
figures for the associated manuscript.

## License
![](https://licensebuttons.net/l/by/3.0/88x31.png)


All creative work (text, plots, images, etc.) are licensed under the
[Creative Commons CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
license. All software is distributed under the standard MIT license as
follows:

```
Copyright (c) 2022 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
