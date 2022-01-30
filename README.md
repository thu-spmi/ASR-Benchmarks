# ASR benchmarks
An effort to track benchmarking results over widely-used datasets for ASR (Automatic Speech Recognition). Note that the ASR results are affected by a number of factors, it is thus important to report results along with those factors for fair comparisons. In this way, we can measure the progress in a more scientifically way. *Feel free to add and correct!*

* [Nomenclature](#Nomenclature)
* [WSJ](#WSJ)
* [Swbd](#Swbd)
* [FisherSwbd](#FisherSwbd)
* [Librispeech](#Librispeech)
* [AISHELL-1](#AISHELL-1)
* [CHiME-4](#CHiME-4)
* [References](#References)

## Nomenclature (alphabetical ordering)
| Terms | Explanations |
| -- |-- |
|**AM** | Acoustic Model. Options: DNN-HMM / CTC / ATT / ATT+CTC / RNN-T / CTC-CRF. Note that: we list some end-to-end (**e2e**) models (e.g., ATT, RNN-T) in this field, although those e2e models contains an implicit/internal LM through the encoder. |
|**AM size (M)** | The number of parameters in millions in the Acoustic Model. Also we report the total number of parameters in e2e models in this field. |
|**ATT**| Attention based Seq2Seq, including **LAS** (Listen Attend and Spell). |
|**CER**| Character Error Rate |
|**Data Aug.**| whether any forms of data augmentations are used, such as **SP** (3-fold Speech Perturbation from Kaldi), **SA** (SpecAugment) |
|**Ext. Data**| whether any forms of external data, such as speech data, text corpus, pretrained models, are used |
|**L**| #Layer, e.g., L24 denotes that the number of layers is 24 |
|**LM**| Language Model, explicitly used, word-level (by default). ''---'' denotes not using shallow fusion with explicit/external LMs, particularly for ATT, RNN-T. |
|**LM size (M)** | The number of parameters in millions in the neural Language Model. For n-gram LMs, this field denotes the total number of n-gram features. |
|**NAS**| Neural Architecture Search|
|**Unit**| phone (namely monophone), biphone, triphone, **wp** (word-piece), character, chenone, **BPE** (byte-pair encoding) |
|**WER**| Word Error Rate |
| --- | not applied |
| ? | not known from the original paper |

## WSJ

This dataset contains about **80 hours** of training data, consisting of read sentences from the Wall Street Journal, recorded under clean conditions. Available from the LDC as WSJ0 under the catalog number [LDC93S6B](https://catalog.ldc.upenn.edu/LDC93S6B).

The evaluation dataset contains the simpler eval92 subset and the harder dev93 subset.

Results are sorted by `eval92` WER.

| eval92 WER | dev93 WER | Unit | AM | AM size (M) | LM | LM size (M) |Data Aug. | Ext. Data | Paper |
| --------- | -------- | - | - | ------- | ----------- | --- | ---- | ---- | ---- |
| 2.50 | 5.48 | mono-phone | CTC-CRF, deformable TDNN | 11.9 | 4-gram | 2.59 |SP | --- | [Deformable TDNN](#deformable) |
| 2.7 | 5.3 | bi-phone | LF-MMI,  TDNN-LSTM | ? | 4-gram | ? |SP | --- | [LF-MMI](#lf-mmi) TASLP2018 |
| 2.77       | 5.68      | mono-phone | CTC-CRF, TDNN NAS | 11.9        | 4-gram                                      | 2.59    | SP        | ---       | [NAS](#st-nas) SLT2021         |
| 3.0        | 6.0      | bi-phone  | EE-LF-MMI, TDNN-LSTM | ?             | 4-gram                                      | ?           | SP        | ---       | [EE-LF-MMI](#lf-mmi) TASLP2018 |
| 3.2        | 5.7      | mono-phone  | CTC-CRF, VGG-BLSTM | 16          | 4-gram                                      | 2.59    | SP        | ---       | [CAT](#cat) IS2020             |
| 3.4        | 5.9       | sub-word| ATT, LSTM | 18            | RNN | 113        | ---       | ---       | [ESPRESSO](#espresso) ASRU2019 |
| 3.79       | 6.23    | mono-phone  | CTC-CRF, BLSTM | 13.5         | 4-gram                                      | 2.59    | SP        | ---       | [CTC-CRF](#ctc-crf) ICASSP2019 |
| 4.9        | ---      | mono-char | ATT+CTC, Transformers | ?           | 4-gram                                      | ?    | SA        | ---       | [phoneBPE-IS2020](#phoneBPE-IS2020) |
| 5.0        | 8.1      | mono-char | CTC-CRF, VGG-BLSTM | 16           | 4-gram                                      | 2.59    | SP        | ---       | [CAT](#cat) IS2020 |

## Swbd

This dataset contains about **260 hours** of English telephone conversations between two strangers on a preassigned topic ([LDC97S62](https://catalog.ldc.upenn.edu/LDC97S62)). The testing is commonly conducted on eval2000 (a.k.a. hub5'00 evaluation, [LDC2002S09](https://catalog.ldc.upenn.edu/LDC2002S09) for speech data and [LDC2002T43](https://catalog.ldc.upenn.edu/LDC2002T43) for transcripts), which consists of two test subsets - Switchboard (SW) and CallHome (CH). 

Results in square brackets denote the weighted average over SW and CH based on our calculation when not reported in the original paper. 

Results are sorted by `Sum` WER.

| SW   | CH   | Sum      | Unit       |AM     | AM size (M) |  LM          | LM size (M) | Data Aug.    | Ext. Data          | Paper                                  |
| :--- | :--- | -------- | :-------------------- | :---------- | :--------- | :---------- | :---------- | ------------ | ------------------ | -------------------------------------- |
| 6.3  | 13.3 | [9.8]      |charBPE &phoneBPE       | ATT+CTC, Transformers, L24 enc, L12 dec | ?  |  multi-level RNNLM        | ?          | SA | Fisher transcripts | [phoneBPE-IS2020](#phoneBPE-IS2020)  |
| 6.4  | 13.4 | 9.9      |char       | RNN-T, BLSTM-LSTM, ivector  | 57          |  LSTM        | 84          | SP, SA, etc. | Fisher transcripts | [Advancing RNN-T](#arnn-t) ICASSP2021  |
| 6.5 | 13.9 | 10.2    |phone   |  LF-MMI, TDNN-f        | ?           | Transformer | 25          | SP           | Fisher transcripts | [P-Rescoring](#p-rescoring) ICASSP2021 |
| 6.8 | 14.1 | [10.5]    |wp 1k   |  ATT               | ?           | LSTM | ?         | SA           | Fisher transcripts | [SpecAug](#SpecAug) IS2019 |
| 6.9 | 14.5 | 10.7 |phone | CTC-CRF     Conformer | 51.82 | Transformer | 25 | SP, SA | Fisher transcripts | [Advancing CTC-CRF](#advancinng-ctc-crf) |
| 7.2 | 14.8 | 11.1 |wp | CTC-CRF  Conformer | 51.85 | Transformer | 25 | SP, SA | Fisher transcripts | [Advancing CTC-CRF](#advancinng-ctc-crf) |
| 7.9  | 15.7 | 11.8     | char      | RNN-T BLSTM-LSTM            | 57          | LSTM        | 5           | SP, SA, etc. | ---                | [Advancing RNN-T](#arnn-t) ICASSP2021  |
| 8.3  | 17.1 | [12.7]  | bi-phone  | LF-MMI, TDNN-LSTM    | ?             | LSTM        | ?           | SP           | Fisher transcripts | [LF-MMI](#lf-mmi) TASLP2018            |
| 8.6  | 17.0 | 12.8    | phone  | LF-MMI, TDNN-f       | ?             | 4-gram      | ?           | SP           | Fisher transcripts | [P-Rescoring](#p-rescoring) ICASSP2021 |
| 8.5  | 17.4 | [13.0]   | bi-phone | EE-LF-MMI, TDNN-LSTM  | ?             | LSTM        | ?           | SP           | Fisher transcripts | [EE-LF-MMI](#lf-mmi) TASLP2018         |
| 8.8  | 17.4 | 13.1    | mono-phone | CTC-CRF, VGG-BLSTM    | 39.2         | LSTM        | ?           | SP           | Fisher transcripts | [CAT](#cat) IS2020 |
| 9.0  | 18.1 | [13.6] | BPE| ATT/CTC | ?  | Transformer | ?  | SP           | Fisher transcripts | [ESPnet-Transformer](#ESPnet-Transformer) ASRU2019 |
| 9.7  | 18.4 | 14.1    | mono-phone | CTC-CRF, chunk-based VGG-BLSTM | 39.2         | 4-gram      | 1.74        | SP           | Fisher transcripts | [CAT](#cat) IS2020                     |
| 9.8  | 18.8 | 14.3     | mono-phone| CTC-CRF, VGG-BLSTM    | 39.2         | 4-gram      | 1.74        | SP           | Fisher transcripts | [CAT](#cat) IS2020                     |
| 10.3 | 19.3 | \[14.8\] | mono-phone| CTC-CRF, BLSTM        | 13.5         | 4-gram      | 1.74        | SP           | Fisher transcripts | [CTC-CRF](#ctc-crf) ICASSP2019         |

## FisherSwbd

The Fisher dataset contains about 1600 hours of English conversational telephone speech (First part: [LDC2004S13](https://catalog.ldc.upenn.edu/LDC2004S13) for speech data, [LDC2004T19](https://catalog.ldc.upenn.edu/LDC2004T19) for transcripts; second part:   [LDC2005S13](https://catalog.ldc.upenn.edu/LDC2005S13) for speech data,  [LDC2005T19](https://catalog.ldc.upenn.edu/LDC2005T19) for transcripts). 

`FisherSwbd` includes both Fisher and Switchboard datasets, which is around **2000 hours** in total. Evaluation is commonly conducted over eval2000 and RT03 ([LDC2007S10](https://catalog.ldc.upenn.edu/LDC2007S10)) datasets.

Results are sorted by `Sum` WER.

| SW   | CH   | Sum    | RT03 | Unit  | AM                   | AM size (M)    | LM   | LM size (M) | Data Aug. | Ext. Data | Paper                          |
| :--- | :--- | ------ | ---- | :------------------- | :---------- | :------- | :--- | :---------- | --------- | --------- | ------------------------------ |
| 7.5  | 14.3 | [10.9] | 10.7 | bi-phone| LF-MMI, TDNN-LSTM    | ?            | LSTM | ?           | SP        | ---       | [LF-MMI](#lf-mmi) TASLP2018    |
| 7.6  | 14.5 | [11.1] | 11.0 | bi-phone| EE-LF-MMI, TDNN-LSTM | ?            | LSTM | ?           | SP        | ---       | [EE-LF-MMI](#lf-mmi) TASLP2018 |
| 7.3  | 15.0 | 11.2 |?    | mono-phone| CTC-CRF, VGG-BLSTM    | 39.2         | LSTM        | ?           | SP           | --- | [CAT](#cat) IS2020 |
| 8.3  | 15.5 | [11.9] |?    | char | ATT  | ? | --- | ---           | SP           | --- | [Tencent-IS2018](#Tencent-IS2018) |
| 8.1  | 17.5 | [12.8] |?    | char | RNN-T | ?  | 4-gram | ? | SP           | --- | [Baidu-ASRU2017](#Baidu-ASRU2017) |

## Librispeech

The LibriSpeech corpus is derived from audiobooks that are part of the LibriVox project, and contains **1000 hours** of speech sampled at 16 kHz. The corpus is freely available for download, along with separately prepared language-model training data and pre-built language models.

There are four test sets: dev-clean, dev-other, test-clean and test-other. For the sake of display, the results are sorted by `test-clean` WER.

| dev clean WER | dev other WER | test clean WER | test other WER | Unit       |AM              | AM size (M) |  LM                                        | LM size (M) | Data Aug. | Ext. Data | Paper              |
| :------------ | :------------ | -------------- | -------------- | :-------------- | :---------- | :--------- | :---------------------------------------- | :---------- | --------- | --------- | ------------------ |
| 1.55          | 4.22          | 1.75           | 4.46          | triphone   | LF-MMI multistream CNN | ?            | self-attentive simple recurrent unit (SRU) | 139            | SA        | ---       | [ASAPP-ASR](#asapp-asr)          |
| 1.7 | 3.6 | 1.8 | 3.6 | wp | CTC Conformer, wav2vec2.0 | 1017 | --- | --- | SA | --- | [ConformerCTC](#conformerctc) |
| ---           | ---           | 1.9            | 3.9          | wp  | RNN-T Conformer | 119          | LSTM                                      | ?           | SA        | Y      | [Conformer](#conformer)          |
| ---           | ---           | 1.9            | 4.1           | wp  | RNN-T ContextNet (L) | 112.7       | LSTM                                      | ?           | SA        | ---       | [ContextNet](#contextnet)         |
| --- | --- | 2.1 | 4.2 | wp | CTC vggTransformer | 81 | Transformer | --- | SP, SA | Y | [FB2020WPM](#fb2020wpm) |
| --- | --- | 2.1 | 4.3 | wp | RNN-T Conformer | 119 | --- | --- | SA | Y | [Conformer](#conformer) |
| --- | --- | 2.26 | 4.85 | chenone | DNN-HMM Transformer | 90 | Transformer | ? | SP, SA | Y | [TransHybrid](#transhybrid) |
| 1.9 | 4.5 | 2.3 | 5.0 | triphone | DNN-HMM BLSTM | ? | Transformer | ? | --- | Y | [RWTH19ASR](#rwth19asr) |
| --- | --- | 2.31 | 4.79 | wp | CTC vggTransformer | 81 | 4-gram | ? | SP, SA | Y | [FB2020WPM](#fb2020wpm) |
| --- | --- | 2.5 | 5.8 | wp | ATT          CNN-BLSTM | ? | RNN | ? | SA | Y | [SpecAug](#SpecAug) IS2019 |
| --- | --- | 2.51 | 5.95 | phone | CTC-CRF Conformer | 51.82 | Transformer | 338 | SA | Y | [Advancing CTC-CRF](#advancinng-ctc-crf) |
| --- | --- | 2.54 | 6.33 | wp | CTC-CRF Conformer | 51.85 | Transformer | 338 | SA | Y | [Advancing CTC-CRF](#advancinng-ctc-crf) |
| --- | --- | 2.6 | 5.59 | chenone | DNN-HMM Transformer | 90 | 4-gram | ? | SP, SA | Y | [TransHybrid](#transhybrid) |
| 2.4 | 5.7 | 2.7 | 5.9 | wp | Conformer | 116 | --- | --- | SA | --- | [ConformerCTC](#conformerctc) |
| --- | --- | 2.8 | 6.8 | wp | ATT          CNN-BLSTM | ? | --- | ? | SA | N | [SpecAug](#SpecAug) IS2019 |
| 2.6 | 8.4 | 2.8 | 9.3 | wp | DNN-HMM LSTM | ? | transformer | ? | --- | Y | [RWTH19ASR](#rwth19asr) |
| 3.87          | 10.28         | 4.09           | 10.65         | phone  | CTC-CRF BLSTM  | 13               | 4-gram                                    | 1.45            | ---       | ---       | [CTC-CRF](#ctc-crf) ICASSP2019|
| ---           | ---           | 4.28           | ---             | tri-phone| LF-MMI   TDNN | ?               | 4-gram                                    | ?            | SP       | ---       | [LF-MMI Interspeech](#lf-mmi-is)|
| 5.1 | 19.1 | 5.9 | 20.0 | biphone | LF-MMI  TDNN-f | ? | 4-gram | ? | SP | Y | [Pkwrap](#pkwrap) |

## AISHELL-1

AISHELL-ASR0009-OS1, is a  **178- hour** open source mandarin speech corpus. It is a part of AISHELL-ASR0009, which contains utterances from 11 domains, including smart home, autonomous driving, and industrial production. The whole recording was made in quiet indoor environment, using 3 different devices at the same time: high fidelity microphone (44.1kHz, 16-bit,); Android-system mobile phone (16kHz, 16-bit), iOS-system mobile phone (16kHz, 16-bit). Audios in high fidelity were re-sampled to 16kHz to build AISHELL- ASR0009-OS1. 400 speakers from different accent areas in China were invited to participate in the recording. The corpus is divided into training, development and testing sets.

| test CER| Unit  | AM                            | AM size (M)      | LM                  | LM size (M) | Data Aug. | Ext. Data | Paper                 |
| :------- | :---------------------------- | :---------- | :-------- | :------------------ | :---------- | --------- | --------- | --------------------- |
| 4.18      | char| RNN-T+CTC, Conformer, LF-MMI | 89           | word 3-gram               | ?           | SA+SP     | ---       | [e2e-word-ngram](#e2e-word-ngram) |
| 4.5      | char| ATT+CTC, Conformer | ?            | LSTM                | ?           | SA+SP     | ---       | [WNARS](#wnars)                 |
| 4.6      | char| ATT+CTC, Transformer | Enc(94.4)+Dec(61.0)+CTC branch(3.3)=158.7            | ---                | ---           | SP     | wav2vec2.0, DistilGPT2       | [preformer-ASRU2021](#preformer-ASRU2021)                 |
| 4.63     | char| ATT+CTC, Conformer | ?            | bidirectional attention rescoring | ?           | SA+SP     | ---       | [U2++](#U2++)                    |
| 4.7     | char| ATT+CTC, Conformer | ?            | Transformer | ?           | SA+SP     | ---       |  [ESPnet-2](https://github.com/espnet/espnet/tree/master/egs2/aishell/asr1#conformer--specaug--speed-perturbation-featsraw-n_fft512-hop_length128) |
| 4.72     | char| ATT+CTC, Conformer | ?            | attention rescoring | ?           | SA+SP     | ---       | [U2](#u2)                    |
| 4.8     | char| RNN-T+CTC, Conformer | 84.3            | --- | ---           | SA+SP     | ---       |  [ESPnet-1](https://github.com/espnet/espnet/blob/master/egs/aishell/asr1/RESULTS.md#conformer-transducer-with-auxiliary-task-ctc-weight--05) |
| 5.2   | char    | Comformer                     | ?           | ---                 | ---           | SA        | ---       | [intermediate CTC loss](#inter-ctc) |
| 6.34     | phone   | CTC-CRF, VGG-BLSTM            | 16           | word 4-gram              | 0.7         | SP        | ---       | [CAT](#cat) IS2020            |


## CHiME-4

The 4th CHiME challenge sets a target for distant-talking automatic speech recognition using a read speech corpus.  Two types of data are employed: 'Real data' - speech data that is recorded in real noisy environments (on a bus, cafe, pedestrian area, and street junction) uttered by actual talkers. 'Simulated data' - noisy utterances that have been generated by artificially mixing clean speech data with noisy backgrounds.

There are four test sets. For the sake of display, the results are sorted by `eval real` WER.

| dev simu WER | dev real WER | eval simu WER | eval real WER | Unit| AM                  | AM size (M)   | LM   | LM size (M) | Data Aug. | Ext. Data | Paper                       |
| :----------- | :----------- | ------------- | ------------- | :------------------ | :---------- | :---- | :--- | :---------- | --------- | --------- | --------------------------- |
| 1.15         | 1.50         | 1.45          | 1.99        | phone  | wide-residual BLSTM | ?            | LSTM | ?           | ---       | ---       | [Complex Spectral Mapping](#complex-spectral-mapping)    |
| 1.78         | 1.69         | 2.12          | 2.24         | phone | 6 DCNN ensemble     | ?            | LSTM | ?           | ---       | ---       | [USTC-iFlytek CHiME4](#ustc-chime4) |
| 2.10         | 1.90         | 2.66          | 2.74        | phone  | LF-MMI, TDNN | ?            | LSTM | ?           | ---       | ---       | [Kaldi-CHiME4](#kaldi-chime4)                |


## References
| Short-hands | Full references |
| :--- | :--- |
| Deformable TDNN<a name="deformable"></a> | Keyu An, Yi Zhang, Zhijian Ou. Deformable TDNN with adaptive receptive fields for speech recognition. Interspeech 2021. |
| CTC-CRF<a name="ctc-crf"></a> ICASSP2019 | H. Xiang, Z. Ou. CRF-based Single-stage Acoustic Modeling with CTC Topology. ICASSP 2019. |
| CAT IS2020<a name="cat"></a> | K. An, H. Xiang, Z. Ou. CAT: A CTC-CRF based ASR Toolkit Bridging the Hybrid and the End-to-end Approaches towards Data Efficiency and Low Latency. INTERSPEECH 2020.|
|NAS<a name="st-nas"></a> SLT2021 | H. Zheng, K. AN, Z. Ou. Efficient Neural Architecture Search for End-to-end Speech Recognition via Straight-Through Gradients. SLT 2021. |
|Conformer<a name="conformer"></a> | Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang. Conformer: Convolution-augmented Transformer for Speech Recognition. INTERSPEECH 2020. |
|ContextNet<a name="contextnet"></a> | Wei Han∗ , Zhengdong Zhang∗ , Yu Zhang, Jiahui Yu, Chung-Cheng Chiu, James Qin, Anmol Gulati, Ruoming Pang, Yonghui Wu. ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context. INTERSPEECH 2020. |
|ASAPP-ASR<a name="asapp-asr"></a> | Jing Pan, Joshua Shapiro, Jeremy Wohlwend, Kyu J. Han, Tao Lei, Tao Ma. ASAPP-ASR: Multistream CNN and Self-Attentive SRU for SOTA Speech Recognition. INTERSPEECH 2020. |
|U2<a name="u2"></a> | Binbin Zhang , Di Wu , Zhuoyuan Yao , Xiong Wang, Fan Yu, Chao Yang, Liyong Guo, Yaguang Hu, Lei Xie , Xin Lei. Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition, arXiv:2012.05481. |
|Kaldi-CHiME4<a name="kaldi-chime4"></a> | Szu-Jui Chen, Aswin Shanmugam Subramanian, Hainan Xu, Shinji Watanabe. Building state-of-the-art distant speech recognition using the CHiME-4 challenge with a setup of speech enhancement baseline. INTERSPEECH 2018. |
|USTC-iFlytek CHiME4  system<a name="ustc-chime4"></a> | Jun Du , Yan-Hui Tu , Lei Sun , Feng Ma , Hai-Kun Wang , Jia Pan , Cong Liu , Jing-Dong Chen , Chin-Hui Lee. The USTC-iFlytek System for CHiME-4 Challenge. |
|Complex Spectral Mapping<a name="complex-spectral-mapping"></a> | Zhong-Qiu Wang,  Peidong Wang, DeLiang Wang. Complex Spectral Mapping for Single- and Multi-Channel Speech Enhancement and Robust ASR. TASLP 2020. |
|intermediate CTC loss<a name="inter-ctc"></a> | Jaesong Lee , Shinji Watanabe. Intermediate Loss Regularization for CTC-based Speech Recognition. ICASSP 2021 |
|WNARS<a name="wnars"></a> | Zhichao Wang, Wenwen Yang, Pan Zhou, Wei Chen. WNARS: WFST based Non-autoregressive Streaming End-to-End Speech Recognition. |
|LF-MMI<a name="lf-mmi"></a> | H. Hadian, H. Sameti, D. Povey, and S. Khudanpur, “Flat-start single-stage discriminatively trained HMM-based models for ASR,” TASLP 2018. |
|LF-MMI Interspeech<a name="lf-mmi-is"></a> | D. Povey, et al. Purely Sequence-Trained Neural Networks for ASR Based on Lattice-Free MMI. Interspeech 2016. |
|ESPRESSO<a name="espresso"></a> | Yiming Wang, Tongfei Chen, Hainan Xu, Shuoyang Ding, Hang Lv, Yiwen Shao, Nanyun Peng, Lei Xie, Shinji Watanabe, and Sanjeev Khudanpur. Espresso: A fast end- to-end neural speech recognition toolkit. ASRU 2019. |
| Advancing RNN-T<a name="arnn-t"></a> | George Saon, Zoltan Tueske, Daniel Bolanos, Brian Kingsbury. Advancing RNN Transducer Technology for Speech Recognition. ICASSP 2021. |
| P-Rescroing<a name="p-rescoring"></a> | Ke Li, Daniel Povey, Sanjeev Khudanpur. A Parallelizable Lattice Rescoring Strategy with Neural Language Models. ICASSP 2021. |
| SpecAug<a name="SpecAug"></a> | D. S. Park, W. Chan, Y. Zhang, et al., SpecAugment: A simple data augmentation method for automatic speech recognition. Interspeech 2019. |
| ESPnet-Transformer<a name="ESPnet-Transformer"></a> | S. Karita, N. Chen, and et al. A comparative study on transformer vs RNN in speech applications,” ASRU 2019.|
| Baidu-ASRU2017<a name="Baidu-ASRU2017"></a> | E. Battenberg, J. Chen, R. Child, A. Coates, Y. Li, H. Liu, S. Satheesh, A. Sriram, and Z. Zhu. Exploring neural transducers for end-to-end speech recognition. ASRU 2017. |
| Tencent-IS2018<a name="Tencent-IS2018"></a> | C. Weng, J. Cui, G. Wang, J. Wang, C. Yu, D. Su, and D. Yu. Improving attention based sequence-to-sequence models for end-to-end English conversational speech recognition. Interspeech 2018. |
| phoneBPE-IS2020<a name="phoneBPE-IS2020"></a> | Weiran Wang, Guangsen Wang, Aadyot Bhatnagar, Yingbo Zhou, Caiming Xiong, and Richard Socher. An investigation of phone-based subword units for end-to-end speech recognition. Interspeech 2020. |
| RWTH19ASR<a name="rwth19asr"></a> | C. Luscher, E. Beck, K. Irie, M. Kitza, W. Michel, A. Zeyer, ¨ R. Schluter, and H. Ney, “RWTH ASR systems for LibriSpeech: Hybrid vs attention-w/o data augmentation," Interspeech 2019. |
| Pkwrap<a name='pkwrap'></a> | Srikanth Madikeri, Sibo Tong, Juan Zuluaga-Gomez, Apoorv Vyas, Petr Motlicek, and Hervé Bourlard, “Pkwrap: a pytorch package for LF-MMI training of acoustic models,” Interspeech 2020. |
| ConformerCTC<a name='conformerctc'></a> | Edwin G Ng, Chung-Cheng Chiu, Yu Zhang, and William Chan. Pushing the limits of non-autoregressive speech recognition. Interspeech 2021. |
| FB2020WPM<a name = 'fb2020wpm'></a> | F. Zhang, Y. Wang, X. Zhang, C. Liu, et al., “Fast, Simpler and More Accurate Hybrid ASR Systems Using Wordpieces,” InterSpeech, 2020. |
| TransHybrid<a name = 'transhybrid'></a> | Yongqiang Wang, Abdelrahman Mohamed, Duc Le, Chunxi Liu, Alex Xiao, Jay Mahadeokar, Hongzhao Huang, Andros Tjandra, Xiaohui Zhang, Frank Zhang, Christian Fuegen, Geoffrey Zweig, and Michael L. Seltzer, “Transformer based acoustic modeling for hybrid speech recognition,” ICASSP 2020. |
| U2++<a name="U2++"></a> | Di Wu, Binbin Zhang, et al. U2++: Unified Two-pass Bidirectional End-to-end Model for Speech Recognition. arXiv:2106.05642. |
| Advancing CTC-CRF<a name="advancing-ctc-crf"></a> | Huahuan Zheng*, Wenjie Peng*, Zhijian Ou, Jinsong Zhang. Advancing CTC-CRF Based End-to-End Speech Recognition with Wordpieces and Conformers. arXiv:2107.03007. |
| e2e-word-ngram<a name="e2e-word-ngram"></a> | Jinchuan Tian, Jianwei Yu, et al. Improving Mandarin End-to-End Speech Recognition with Word N-gram Language Model. arXiv:2201.01995. |
