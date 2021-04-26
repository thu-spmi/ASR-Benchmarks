# ASR benchmarks
An effort to track benchmarking results over widely-used datasets for ASR (Automatic Speech Recognition). Note that the ASR results are affected by a number of factors, it is thus important to report results along with those factors for fair comparisons. In this way, we can measure the progress in a more scientifically way. *Feel free to add and correct!*

[TOC]
## Nomenclature
| Terms | Explanations |
| -- |-- |
|**AM** | Acoustic Model. Note that: we also list the end-to-end (**e2e**) models (e.g., Attention based Seq2Seq, RNN-T) in this field, although these e2e models contains an implicit/internal LM through the encoder. |
|**AM size (M)** | The number of parameters in millions in the Acoustic Model. Also we report the total number of parameters in the e2e models in this field. |
|**Unit**| phone (namely monophone), biphone, triphone, **wp** (word-piece), character |
|**LM**| Language Model, explicitly used. ''---'' denotes not using shallow fusion with explicit/external LMs, particularly for Attention based Seq2Seq, RNN-T. |
|**LM size (M)** | The number of parameters in millions in the neural Language Model. For n-gram LMs, this field denotes the total number of n-gram features. |
|**Data Aug.**| whether any forms of data augmentations are used, such as **SP** (3-fold Speech Perturbation from Kaldi), **SA** (SpecAugment) |
|**Ext. Data**| whether any forms of external data (either speech data or text corpus) are used |
|**NAS**| Neural Architecture Search|
|**WER**| Word Error Rate |
|**CER**| Character Error Rate |
| --- | not applied |
| ? | not known from the original paper |


## WSJ

This dataset contains about **80 hours** of training data, consisting of read sentences from the Wall Street Journal, recorded under clean conditions. Available from the LDC as WSJ0 under the catalog number [LDC93S6B](https://catalog.ldc.upenn.edu/LDC93S6B).

The evaluation dataset contains the simpler eval92 subset and the harder dev93 subset.

Results are sorted by `eval92` WER.

| eval92 WER | dev93 WER | AM | AM size (M) | Unit | LM | LM size (M) |Data Aug. | Ext. Data | Paper |
| --------- | -------- | - | - | ------- | ----------- | --- | ---- | ---- | ---- |
| 2.7 | 5.3 | TDNN-LSTM | ? | bi-phone | 4-gram | ? |SP | --- | [LF-MMI](#lf-mmi) TASLP2018 |
| 2.77       | 5.68      | TDNN NAS  | 11.9        | mono-phone | 4-gram                                      | 0.18        | SP        | ---       | [NAS](#st-nas) SLT2021         |
| 3.0        | 6.0       | TDNN-LSTM | ?           | bi-phone   | 4-gram                                      | ?           | SP        | ---       | [EE-LF-MMI](#lf-mmi) TASLP2018 |
| 3.2        | 5.7       | VGG BLSTM | 16          | mono-phone | 4-gram                                      | 0.18        | SP        | ---       | [CAT](#cat) IS2020             |
| 3.4        | 5.9       | LSTM      | ?           | sub-word   | CNN-LSTM encoder and LSTM/attention decoder | ?           | ---       | ---       | [ESPRESSO](#espresso) ASRU2019 |
| 3.79       | 6.23      | BLSTM     | 13.5        | mono-phone | 4-gram                                      | 0.18        | SP        | ---       | [CTC-CRF](#ctc-crf) ICASSP2019 |

## Swbd

This dataset contains about **260 hours** of English telephone conversations between two strangers on a preassigned topic ([LDC97S62](https://catalog.ldc.upenn.edu/LDC97S62)). The testing is commonly conducted on eval2000 (a.k.a. hub5'00 evaluation, [LDC2002S09](https://catalog.ldc.upenn.edu/LDC2002S09) for speech data and [LDC2002T43](https://catalog.ldc.upenn.edu/LDC2002T43) for transcripts), which consists of two test subsets - Switchboard (SW) and CallHome (CH). 

Results in square brackets denote the weighted average over SW and CH based on our calculation when not reported in the original paper. 

Results are sorted by `Sum` WER.

| SW   | CH   | Sum      | AM                    | AM size (M) | Unit       | LM          | LM size (M) | Data Aug.    | Ext. Data          | Paper                                  |
| :--- | :--- | -------- | :-------------------- | :---------- | :--------- | :---------- | :---------- | ------------ | ------------------ | -------------------------------------- |
| 6.4  | 13.4 | 9.9      | BLSTM-LSTM            | 57          | char       | LSTM        | 84          | SP, SA, etc. | Fisher transcripts | [Advancing RNN-T](#arnn-t) ICASSP2021  |
| 7.2  | 14.4 | 10.8     | TDNN-f                | ?           | Word       | Transformer | 25          | SP           | Fisher transcripts | [P-Rescoring](#p-rescoring) ICASSP2021 |
| 7.9  | 15.7 | 11.8     | BLSTM-LSTM            | 57          | Char       | LSTM        | 5           | SP, SA, etc. | ---                | [Advancing RNN-T](#arnn-t) ICASSP2021  |
| 8.3  | 17.1 | [12.7]   | TDNN-LSTM             | ?           | bi-phone   | LSTM        | ?           | SP           | Fisher transcripts | [LF-MMI](#lf-mmi) TASLP2018            |
| 8.6  | 17.0 | 12.8     | TDNN-f                | ?           | Word       | 4-gram      | ?           | SP           | Fisher transcripts | [P-Rescoring](#p-rescoring) ICASSP2021 |
| 8.5  | 17.4 | [13.0]   | TDNN-LSTM             | ?           | bi-phone   | LSTM        | ?           | SP           | Fisher transcripts | [EE-LF-MMI](#lf-mmi) TASLP2018         |
| 8.8  | 17.4 | 13.1     | VGG BLSTM             | 39.2        | Mono-phone | LSTM        | ?           | SP           | Fisher transcripts | [CAT](#cat) IS2020                     |
| 9.7  | 18.4 | 14.1     | chunk-based VGG BLSTM | 39.2        | Mono-phone | 4-gram      | 1.74        | SP           | Fisher transcripts | [CAT](#cat) IS2020                     |
| 9.8  | 18.8 | 14.3     | VGG BLSTM             | 39.2        | Mono-phone | 4-gram      | 1.74        | SP           | Fisher transcripts | [CAT](#cat) IS2020                     |
| 10.3 | 19.3 | \[14.8\] | BLSTM                 | 13.5        | Mono-phone | 4-gram      | 1.74        | SP           | Fisher transcripts | [CTC-CRF](#ctc-crf) ICASSP2019         |

## FisherSwbd

The Fisher dataset contains about 1600 hours of English conversational telephone speech (First part: [LDC2004S13](https://catalog.ldc.upenn.edu/LDC2004S13) for speech data, [LDC2004T19](https://catalog.ldc.upenn.edu/LDC2004T19) for transcripts; second part:   [LDC2005S13](https://catalog.ldc.upenn.edu/LDC2005S13) for speech data,  [LDC2005T19](https://catalog.ldc.upenn.edu/LDC2005T19) for transcripts). 

`FisherSwbd` includes both Fisher and Switchboard datasets, which is arount 2000 hours in total. Evaluation is commonly conducted over eval2000 and RT03 ([LDC2007S10](https://catalog.ldc.upenn.edu/LDC2007S10)) datasets.

Results are sorted by `Sum` WER.

| SW   | CH   | Sum    | RT03 | AM        | AM size (M) | Unit     | LM   | LM size (M) | Data Aug. | Ext. Data | Paper                          |
| :--- | :--- | ------ | ---- | :-------- | :---------- | :------- | :--- | :---------- | --------- | --------- | ------------------------------ |
| 7.5  | 14.3 | [10.9] | 10.7 | TDNN-LSTM | ?           | bi-phone | LSTM | ?           | SP        | ---       | [LF-MMI](#lf-mmi) TASLP2018    |
| 7.6  | 14.5 | [11.1] | 11.0 | TDNN-LSTM | ?           | bi-phone | LSTM | ?           | SP        | ---       | [EE-LF-MMI](#lf-mmi) TASLP2018 |

## Librispeech

The LibriSpeech corpus is derived from audiobooks that are part of the LibriVox project, and contains **1000 hours** of speech sampled at 16 kHz. The corpus is freely available for download, along with separately prepared language-model training data and pre-built language models.

There are four test sets. For the sake of display, the better for test clean, the earlier in the following Table.

| dev clean WER | dev other WER | test clean WER | test other WER | AM              | AM size (M) | Unit       | LM                                        | LM size (M) | Data Aug. | Ext. Data | Paper              |
| :------------ | :------------ | -------------- | -------------- | :-------------- | :---------- | :--------- | :---------------------------------------- | :---------- | --------- | --------- | ------------------ |
| 1.55          | 4.22          | 1.75           | 4.46           | multistream CNN | ?           | triphone   | selfattentive simple recurrent unit (SRU) | 139            | SA        | ---       | [ASAPP-ASR](#asapp-asr)          |
| ---           | ---           | 1.9            | 3.9            | Conformer       | 119         | word piece | LSTM                                      | ?           | SA        | ---       | [Conformer](#conformer)          |
| ---           | ---           | 1.9            | 4.1            | ContextNet (L)  | 112.7       | word piece | LSTM                                      | ?           | SA        | ---       | [ContextNet](#contextnet)         |
| 3.87          | 10.28         | 4.09           | 10.65          | BLSTM           | 13          | phone      | 4-gram                                    | 1.45            | ---       | ---       | [CTC-CRF](#ctc-crf) ICASSP2019|


## AISHELL-1

AISHELL-ASR0009-OS1, is a  **178**- hour open source mandarin speech corpus. It is a part of AISHELL-ASR0009, of which utterance contains 11 domains, including smart home, autonomous driving, and industrial production. The whole recording was put in quiet indoor environment, using 3 different devices at the same time: high fidelity microphone (44.1kHz, 16-bit,); Android-system mobile phone (16kHz, 16-bit), iOS-system mobile phone (16kHz, 16-bit). Audios in high fidelity were re-sampled to 16kHz to build AISHELL- ASR0009-OS1. 400 speakers from different accent areas in China were invited to participate in the recording. The corpus is divided into training, development and testing sets.

| test CER | AM                            | AM size (M) | Unit      | LM                  | LM size (M) | Data Aug. | Ext. Data | Paper                 |
| :------- | :---------------------------- | :---------- | :-------- | :------------------ | :---------- | --------- | --------- | --------------------- |
| 4.5      | Conformer based CTC/attention | ?           | character | LSTM                | ?           | SA+SP     | ---       | [WNARS](#wnars)                 |
| 4.72     | Conformer based CTC/attention | ?           | character | attention rescoring | ?           | SA+SP     | ---       | [U2](#u2)                    |
| 5.2      | Comformer                     | ?           | character | ---                 | ?           | SA        | ---       | [intermediate CTC loss](#inter-ctc) |
| 6.34     | VGGBLSTM                      | 16M         | phone     | 4-gram              | 0.7         | SP        | ---       | [CAT](#cat) IS2020            |


## CHiME-4

The 4th CHiME challenge sets a target for distant-talking automatic speech recognition using a read speech corpus.  Two types of data are employed: 'Real data' - speech data that is recorded in real noisy environments (on a bus, cafe, pedestrian area, and street junction) uttered by actual talkers. 'Simulated data' - noisy utterances that have been generated by artificially mixing clean speech data with noisy backgrounds.

There are four test sets. For the sake of display, the better for eval real, the earlier in the following Table.

| dev simu WER | dev real WER | eval simu WER | eval real WER | AM                  | AM size (M) | Unit  | LM   | LM size (M) | Data Aug. | Ext. Data | Paper                       |
| :----------- | :----------- | ------------- | ------------- | :------------------ | :---------- | :---- | :--- | :---------- | --------- | --------- | --------------------------- |
| 1.15         | 1.50         | 1.45          | 1.99          | wide-residual BLSTM | ?           | phone | LSTM | ?           | ---       | ---       | [Complex Spectral Mapping](#complex-spectral-mapping)    |
| 1.78         | 1.69         | 2.12          | 2.24          | 6 DCNN ensemble     | ?           | phone | LSTM | ?           | ---       | ---       | [USTC-iFlytek CHiME4](#ustc-chime4)  system |
| 2.10         | 1.90         | 2.66          | 2.74          | TDNN with LF-MMI    | ?           | phone | LSTM | ?           | ---       | ---       | [Kaldi-CHiME4](#kaldi-chime4)                |


## References
| Short-hands | Full references |
| :--- | :--- |
| CTC-CRF<a name="ctc-crf"></a> ICASSP2019 | H. Xiang, Z. Ou. CRF-based Single-stage Acoustic Modeling with CTC Topology. ICASSP, 2019. |
| CAT IS2020<a name="cat"></a> | K. An, H. Xiang, Z. Ou. CAT: A CTC-CRF based ASR Toolkit Bridging the Hybrid and the End-to-end Approaches towards Data Efficiency and Low Latency. INTERSPEECH, 2020.|
|NAS<a name="st-nas"></a> SLT2021 | H. Zheng, K. AN, Z. Ou. Efficient Neural Architecture Search for End-to-end Speech Recognition via Straight-Through Gradients. SLT 2021.|
|Conformer<a name="conformer"></a> | Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang. Conformer: Convolution-augmented Transformer for Speech Recognition. INTERSPEECH 2020. |
|ContextNet<a name="contextnet"></a> | Wei Han∗ , Zhengdong Zhang∗ , Yu Zhang, Jiahui Yu, Chung-Cheng Chiu, James Qin, Anmol Gulati, Ruoming Pang, Yonghui Wu. ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context. INTERSPEECH 2020. |
|ASAPP-ASR<a name="asapp-asr"></a> | Jing Pan, Joshua Shapiro, Jeremy Wohlwend, Kyu J. Han, Tao Lei, Tao Ma. ASAPP-ASR: Multistream CNN and Self-Attentive SRU for SOTA Speech Recognition. INTERSPEECH 2020. |
|U2<a name="u2"></a> | Binbin Zhang , Di Wu , Zhuoyuan Yao , Xiong Wang, Fan Yu, Chao Yang, Liyong Guo, Yaguang Hu, Lei Xie , Xin Lei. Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition. |
|Kaldi-CHiME4<a name="kaldi-chime4"></a> | Szu-Jui Chen, Aswin Shanmugam Subramanian, Hainan Xu, Shinji Watanabe. Building state-of-the-art distant speech recognition using the CHiME-4 challenge with a setup of speech enhancement baseline. INTERSPEECH 2018. |
|USTC-iFlytek CHiME4  system<a name="ustc-chime4"></a> | Jun Du , Yan-Hui Tu , Lei Sun , Feng Ma , Hai-Kun Wang , Jia Pan , Cong Liu , Jing-Dong Chen , Chin-Hui Lee. The USTC-iFlytek System for CHiME-4 Challenge. |
|Complex Spectral Mapping<a name="complex-spectral-mapping"></a> | Zhong-Qiu Wang,  Peidong Wang , DeLiang Wang. Complex Spectral Mapping for Single- and Multi-Channel Speech Enhancement and Robust ASR. TASLP 2020. |
|intermediate CTC loss<a name="inter-ctc"></a> | Jaesong Lee , Shinji Watanabe. Intermediate Loss Regularization for CTC-based Speech Recognition. ICASSP 2021 |
|WNARS<a name="wnars"></a> | Zhichao Wang, Wenwen Yang, Pan Zhou, Wei Chen. WNARS: WFST based Non-autoregressive Streaming End-to-End Speech Recognition. |
|LF-MMI<a name="lf-mmi"></a> | H. Hadian, H. Sameti, D. Povey, and S. Khudanpur, “Flat- start single-stage discriminatively trained HMM-based models for ASR,” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2018. |
|ESPRESSO<a name="espresso"></a> | Yiming Wang, Tongfei Chen, Hainan Xu, Shuoyang Ding, Hang Lv, Yiwen Shao, Nanyun Peng, Lei Xie, Shinji Watanabe, and Sanjeev Khudanpur, “Espresso: A fast end- to-end neural speech recognition toolkit,” in *ASRU*, 2019. |
| ARNN-T<a name="arnn-r"></a> | George Saon, Zoltan Tueske, Daniel Bolanos, Brian Kingsbury. Advancing RNN Transducer Technology for Speech Recognition. ICASSP, 2021. |
| P-Rescroing<a name="p-rescoring"></a> | Ke Li, Daniel Povey, Sanjeev Khudanpur. A Parallelizable Lattice Rescoring Strategy with Neural Language Models. ICASSP, 2021. |
