# Third-Party Notices

This project builds on and adapts code from several open-source projects.
Each is listed below with its origin, license, and the files it covers in
this repository. Full upstream license texts are reproduced at the bottom
of this file.

| Files in this repo | Upstream project | Authors | License |
|---|---|---|---|
| Overall repo scaffold, `lib/` package layout, `experiments/run_real.py` (AQI setup) | [andreacini/sparse-graph-learning](https://github.com/andreacini/sparse-graph-learning) (JMLR 2023) | Andrea Cini, Daniele Zambon, Cesare Alippi | MIT |
| `lib/nn/graph_samplers/imle/{imle,noise,solver}.py` | [uclnlp/torch-imle](https://github.com/uclnlp/torch-imle) | Mathias Niepert, Pasquale Minervini, Luca Franceschi | MIT |
| `lib/nn/graph_samplers/imle/{aimle,target_dist}.py` | [EdinburghNLP/torch-adaptive-imle](https://github.com/EdinburghNLP/torch-adaptive-imle) | Pasquale Minervini, Luca Franceschi, Mathias Niepert | MIT |
| `lib/nn/graph_samplers/bes.py` (`ConcreteBinarySampler`) | [yaringal/ConcreteDropout](https://github.com/yaringal/ConcreteDropout) | Yarin Gal | MIT |
| `lib/utils/distributions.py` | [wouterkool/estimating-gradients-without-replacement](https://github.com/wouterkool/estimating-gradients-without-replacement) | Wouter Kool, Herke van Hoof, Max Welling | Unlicensed upstream; reused here under the same attribution convention as the original Cini et al. (2023) repository, which adapted this same file with the same notice under its own MIT license. |
| `lib/gradient_estimators/doublecv.py` | [thjashin/double-cv](https://github.com/thjashin/double-cv) (reimplementation based on the paper, not a line-for-line port) | Michalis K. Titsias, Jiaxin Shi | MIT |
| `lib/nn/satorras_forecaster.py` | Reimplementation based on the architecture described in Satorras, Rangapuram & Januschowski, "Multivariate Time Series Forecasting with Latent Graph Inference" (2022). No official code release was found at the time of writing; no code was copied. | Victor Garcia Satorras, Syama Sundar Rangapuram, Tim Januschowski | N/A (paper-based reimplementation) |

All other code in this repository (predictors, datasets, training loops,
experiment configs not listed above) is original work.

---

## MIT License (uclnlp/torch-imle, EdinburghNLP/torch-adaptive-imle,
## yaringal/ConcreteDropout, thjashin/double-cv, andreacini/sparse-graph-learning)

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
