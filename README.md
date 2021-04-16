# ASR benchmarks
An effort to track benchmarking results over widely-used datasets for ASR (Automatic Speech Recognition). Note that the ASR results are affected by a number of factors, it is thus important to report results along with those factors for fair comparisons. In this way, we can measure the progress in a more scientifically way. *Feel free to add and correct!*

[TOC]
## Nomenclature
| Terms | Explanations |
| :-- |: -- |
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

The evaluation dataset contains the simpler eval92 subset and the harder dev93 subset. For the sake of display, the better for eval92, the earlier in the following Table.

| eval92 WER | dev93 WER | AM | AM size (M) | Unit | LM | LM size (M) |Data Aug. | Ext. Data | Paper |
| :--------- | :-------- | :- | :- | :------- | :----------- | :--- | ---- | ---- | ---- |
| 3.79 | 6.23 | BLSTM |13| mono-phone | 4-gram |?| SP |--- | [CTC-CRF](#ctc-crf) ICASSP2019 |
| 3.2 | 5.7 | VGG BLSTM |16| mono-phone | 4-gram |?| SP |--- | [CAT](#cat) IS2020 |
| 2.77 | 5.68 | TDNN NAS |11.9| mono-phone | 4-gram |?| SP |--- | NAS SLT2021|
| 2.7 | 5.3 |  || bi-phone | 4-gram ||  | | LF-MMI ASLP18 |
| 3.0 | 6.0 |  || bi-phone | 4-gram ||  | | EE-LF-MMI ASLP18 |
|  |  |  ||  |  ||  | |  |

## Swbd

This dataset contains about **260 hours** of English telephone conversations between two strangers on a preassigned topic. The testing is commonly conducted on eval2000 (a.k.a. hub5'00 evaluation), which consists of two test subsets - Switchboard (SW) and CallHome (CH). Results in square brackets denote the weighted average over SW and CH based on our calculation when not reported in the original paper. 

| SW   | CH   | Sum      | AM                    | AM size (M) | Unit       | LM     | LM size (M) | Data Aug. | Ext. Data | Paper              |
| :--- | :--- | -------- | :-------------------- | :---------- | :--------- | :----- | :---------- | --------- | --------- | ------------------ |
| 10.3 | 19.3 | \[15.0\] | BLSTM                 | 13.47       | Mono-phone | 4-gram |             | SP        | ---       | CTC-CRF ICASSP2019 |
| 9.8  | 18.8 | 14.3     | VGG BLSTM             | 39.15       | Mono-phone | 4-gram |             | SP        | ---       | CAT IS2020         |
| 8.8  | 17.4 | 13.1     | VGG BLSTM             | 39.15       | Mono-phone | LSTM   |             | SP        | ---       | CAT IS2020         |
| 9.7  | 18.4 | 14.1     | chunk-based VGG BLSTM | 39.15       | Mono-phone | 4-gram |             | SP        | ---       | CAT IS2020         |
|      |      |          |                       |             |            |        |             |           |           |                    |
|      |      |          |                       |             |            |        |             |           |           |                    |

## FisherSwbd

## Librispeech

The LibriSpeech corpus is derived from audiobooks that are part of the LibriVox project, and contains **1000 hours** of speech sampled at 16 kHz. The corpus is freely available for download, along with separately prepared language-model training data and pre-built language models.

There are four test sets. For the sake of display, the better for test clean, the earlier in the following Table.

| dev clean WER | dev other WER | test clean WER | test other WER | AM              | AM size (M) | Unit       | LM                                        | LM size (M) | Data Aug. | Ext. Data | Paper              |
| :------------ | :------------ | -------------- | -------------- | :-------------- | :---------- | :--------- | :---------------------------------------- | :---------- | --------- | --------- | ------------------ |
| 3.87          | 10.28         | 4.09           | 10.65          | BLSTM           | 13          | phone      | 4-gram                                    |             | ---       | ---       | CTC-CRF ICASSP2019 |
| ---           | ---           | 1.9            | 3.9            | Conformer       | 118         | word piece | LSTM                                      |             | SA        | ---       | Conformer          |
| 1.55          | 4.22          | 1.75           | 4.46           | multistream CNN |             | triphone   | selfattentive simple recurrent unit (SRU) |             | SA        | ---       | ASAPP-ASR          |
| ---           | ---           | 1.9            | 4.1            | ContextNet (L)  | 112.7       | word piece | LSTM                                      |             | SA        | ---       | ContextNet         |

## AISHELL-1

AISHELL-ASR0009-OS1, is a  **178**- hour open source mandarin speech corpus. It is a part of AISHELL-ASR0009, of which utterance contains 11 domains, including smart home, autonomous driving, and industrial production. The whole recording was put in quiet indoor environment, using 3 different devices at the same time: high fidelity microphone (44.1kHz, 16-bit,); Android-system mobile phone (16kHz, 16-bit), iOS-system mobile phone (16kHz, 16-bit). Audios in high fidelity were re-sampled to 16kHz to build AISHELL- ASR0009-OS1. 400 speakers from different accent areas in China were invited to participate in the recording. The corpus is divided into training, development and testing sets.

| test CER | AM                            | AM size (M) | Unit      | LM                  | LM size (M) | Data Aug. | Ext. Data | Paper                 |
| :------- | :---------------------------- | :---------- | :-------- | :------------------ | :---------- | --------- | --------- | --------------------- |
| 6.34     | VGGBLSTM                      | 16M         | phone     | 4-gram              |             | SP        | ---       | CAT IS2020            |
| 4.72     | Conformer based CTC/attention |             | character | attention rescoring |             | SA+SP     | ---       | U2                    |
| 5.2      | Comformer                     |             | character | ---                 |             | SA        | ---       | intermediate CTC loss |
| 4.5      | Conformer based CTC/attention |             | character | LSTM                |             | SA+SP     | ---       | WNARS                 |

## CHiME-4

The 4th CHiME challenge sets a target for distant-talking automatic speech recognition using a read speech corpus.  Two types of data are employed: 'Real data' - speech data that is recorded in real noisy environments (on a bus, cafe, pedestrian area, and street junction) uttered by actual talkers. 'Simulated data' - noisy utterances that have been generated by artificially mixing clean speech data with noisy backgrounds.

There are four test sets. For the sake of display, the better for eval real, the earlier in the following Table.

| dev simu WER | dev real WER | eval simu WER | eval real WER | AM                  | AM size (M) | Unit  | LM   | LM size (M) | Data Aug. | Ext. Data | Paper                       |
| :----------- | :----------- | ------------- | ------------- | :------------------ | :---------- | :---- | :--- | :---------- | --------- | --------- | --------------------------- |
| 1.78         | 1.69         | 2.12          | 2.24          | 6 DCNN ensemble     |             | phone | LSTM |             | ---       | ---       | USTC-iFlytek CHiME4  system |
| 2.10         | 1.90         | 2.66          | 2.74          | TDNN with LF-MMI    |             | phone | LSTM |             | ---       | ---       | Kaldi-CHiME4                |
| 1.15         | 1.50         | 1.45          | 1.99          | wide-residual BLSTM |             | phone | LSTM |             | ---       | ---       | Complex Spectral Mapping    |
|              |              |               |               |                     |             |       |      |             |           |           |                             |

## References
| Short-hands | Full references |
| :--- | :--- |
| CTC-CRF<a name="ctc-crf"></a> ICASSP2019 | H. Xiang, Z. Ou. CRF-based Single-stage Acoustic Modeling with CTC Topology. ICASSP, 2019. |
| CAT IS2020<a name="cat"></a> | K. An, H. Xiang, Z. Ou. CAT: A CTC-CRF based ASR Toolkit Bridging the Hybrid and the End-to-end Approaches towards Data Efficiency and Low Latency. INTERSPEECH, 2020.|
|NAS SLT2021 | H. Zheng, K. AN, Z. Ou. Efficient Neural Architecture Search for End-to-end Speech Recognition via Straight-Through Gradients. SLT 2021.|
|Conformer | Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang. Conformer: Convolution-augmented Transformer for Speech Recognition. INTERSPEECH 2020. |
|ContextNet | Wei Han∗ , Zhengdong Zhang∗ , Yu Zhang, Jiahui Yu, Chung-Cheng Chiu, James Qin, Anmol Gulati, Ruoming Pang, Yonghui Wu. ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context. INTERSPEECH 2020. |
|ASAPP-ASR | Jing Pan, Joshua Shapiro, Jeremy Wohlwend, Kyu J. Han, Tao Lei, Tao Ma. ASAPP-ASR: Multistream CNN and Self-Attentive SRU for SOTA Speech Recognition. INTERSPEECH 2020. |
|U2 | Binbin Zhang , Di Wu , Zhuoyuan Yao , Xiong Wang, Fan Yu, Chao Yang, Liyong Guo, Yaguang Hu, Lei Xie , Xin Lei. Unified Streaming and Non-streaming Two-pass End-to-end Model for Speech Recognition. |
|Kaldi-CHiME4 | Szu-Jui Chen, Aswin Shanmugam Subramanian, Hainan Xu, Shinji Watanabe. Building state-of-the-art distant speech recognition using the CHiME-4 challenge with a setup of speech enhancement baseline. INTERSPEECH 2018. |
|USTC-iFlytek CHiME4  system | Jun Du , Yan-Hui Tu , Lei Sun , Feng Ma , Hai-Kun Wang , Jia Pan , Cong Liu , Jing-Dong Chen , Chin-Hui Lee. The USTC-iFlytek System for CHiME-4 Challenge. |
|Complex Spectral Mapping | Zhong-Qiu Wang,  Peidong Wang , DeLiang Wang. Complex Spectral Mapping for Single- and Multi-Channel Speech Enhancement and Robust ASR. TASLP 2020. |
|intermediate CTC loss | Jaesong Lee , Shinji Watanabe. Intermediate Loss Regularization for CTC-based Speech Recognition. ICASSP 2021 |
|WNARS | Zhichao Wang, Wenwen Yang, Pan Zhou, Wei Chen. WNARS: WFST based Non-autoregressive Streaming End-to-End Speech Recognition. |

