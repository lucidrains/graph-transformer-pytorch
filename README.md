## Graph Transformer - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2009.03509">Graph Transformer</a> in Pytorch, for potential use in replicating <a href="https://github.com/lucidrains/alphafold2">Alphafold2</a>. This was recently used by both <a href="https://www.biorxiv.org/content/10.1101/2021.06.02.446809v1">Costa et al</a> and <a href="https://www.biorxiv.org/content/10.1101/2021.06.14.448402v1">Bakers lab</a> for transforming MSA and pair-wise embedding into 3d coordinates.

## Citations

```bibtex
@article {Costa2021.06.02.446809,
    author  = {Costa, Allan and Ponnapati, Manvitha and Jacobson, Joseph M. and Chatterjee, Pranam},
    title   = {Distillation of MSA Embeddings to Folded Protein Structures with Graph Transformers},
    year    = {2021},
    doi     = {10.1101/2021.06.02.446809},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/early/2021/06/02/2021.06.02.446809},
    eprint  = {https://www.biorxiv.org/content/early/2021/06/02/2021.06.02.446809.full.pdf},
    journal = {bioRxiv}
}
```

```bibtex
@article {Baek2021.06.14.448402,
    author  = {Baek, Minkyung and DiMaio, Frank and Anishchenko, Ivan and Dauparas, Justas and Ovchinnikov, Sergey and Lee, Gyu Rie and Wang, Jue and Cong, Qian and Kinch, Lisa N. and Schaeffer, R. Dustin and Mill{\'a}n, Claudia and Park, Hahnbeom and Adams, Carson and Glassman, Caleb R. and DeGiovanni, Andy and Pereira, Jose H. and Rodrigues, Andria V. and van Dijk, Alberdina A. and Ebrecht, Ana C. and Opperman, Diederik J. and Sagmeister, Theo and Buhlheller, Christoph and Pavkov-Keller, Tea and Rathinaswamy, Manoj K and Dalwadi, Udit and Yip, Calvin K and Burke, John E and Garcia, K. Christopher and Grishin, Nick V. and Adams, Paul D. and Read, Randy J. and Baker, David},
    title   = {Accurate prediction of protein structures and interactions using a 3-track network},
    year    = {2021},
    doi     = {10.1101/2021.06.14.448402},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/early/2021/06/15/2021.06.14.448402},
    eprint  = {https://www.biorxiv.org/content/early/2021/06/15/2021.06.14.448402.full.pdf},
    journal = {bioRxiv}
}
```

```bibtex
@misc{shi2021masked,
    title   = {Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification}, 
    author  = {Yunsheng Shi and Zhengjie Huang and Shikun Feng and Hui Zhong and Wenjin Wang and Yu Sun},
    year    = {2021},
    eprint  = {2009.03509},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
