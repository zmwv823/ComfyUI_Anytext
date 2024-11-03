---
title: Glyph-SDXL-v2
emoji: 🖼️🖌️
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 4.31.1
app_file: app.py
pinned: false
---
# Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering
[Glyph-ByT5 Project Page](https://glyph-byt5.github.io/) |[Glyph-ByT5-v2 Project Page](https://glyph-byt5-v2.github.io/) | [Glyph-ByT5 arXiv Paper](https://arxiv.org/abs/2403.09622) |[Glyph-ByT5-v2 arXiv Paper](https://arxiv.org/abs/2406.10208) | [Github](https://github.com/AIGText/Glyph-ByT5)
#### We present a basic version of Glyph-SDXL, and a multilingual version Glyph-SDXL-v2 supporting up to 10 languages: English, Chinese, French, German, Spanish, Portuguese, Italian, Russian, Japanese and Korean.

#### Note: due to limited capacity, we support 5000 chars in Chinese, 1148 chars in Japanese and 617 in Korean. Certain uncommon characters might not be supported for these three languages.

#### Models presented in this demo are all based on albedo-xl!
# 🚀🚀🚀 🔥🔥🔥 Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering


<a href='https://arxiv.org/abs/2403.09622'><img src='https://img.shields.io/badge/Arxiv-2403.09622-red'>
<a href='https://arxiv.org/abs/2406.10208'><img src='https://img.shields.io/badge/Arxiv-2406.10208-red'>
<a href='https://glyph-byt5.github.io/'><img src='https://img.shields.io/badge/Project Page-GlyphByT5-green'>
<a href='https://glyph-byt5-v2.github.io/'><img src='https://img.shields.io/badge/Project Page-GlyphByT5v2-green'></a>
<a href='https://huggingface.co/datasets/GlyphByT5/GlyphByT5Pretraining'><img src='https://img.shields.io/badge/Dataset-GlyphByT5Pretraining-yellow'>


This is the official implementation of Glyph-ByT5 and Glyph-ByT5-v2, introduced in [Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering](https://arxiv.org/abs/2403.09622) and [Glyph-ByT5-v2: A Strong Aesthetic Baseline for Accurate Multilingual Visual Text Rendering
](https://arxiv.org/abs/2406.10208). 

## News

⛽⛽⛽  Contact: [yuhui.yuan@microsoft.com](yuhui.yuan@microsoft.com)

**2024.06.28** We have removed the weights and code that may have used potentially unauthorized datasets in the current stage. We will update the checkpoints after the Microsoft RAI process.

## :high_brightness: Highlights

* We identify two crucial requirements of text encoders for achieving accurate visual text rendering: character awareness and alignment with glyphs. To this end, we propose a customized text encoder, Glyph-ByT5, by fine-tuning the character-aware ByT5 encoder using a meticulously curated paired glyph-text dataset.

* We present an effective method for integrating Glyph-ByT5 with SDXL, resulting in the creation of the Glyph-SDXL model for design image generation. This significantly enhances text rendering accuracy, improving it from less than 20% to nearly 90% on our design image benchmark. Noteworthy is Glyph-SDXL's newfound ability for text paragraph rendering, achieving high spelling accuracy for tens to hundreds of characters with automated multi-line layouts.

* We deliver a powerful customized multilingual text encoder, Glyph-ByT5-v2, and a strong aesthetic graphic generation model, Glyph-SDXL-v2, that can support accurate spelling in $\sim10$ different languages

<table>
  <tr>
    <td><img src="inference/assets/teaser/paragraph_1.png" alt="paragraph example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/paragraph_2.png" alt="paragraph example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/paragraph_3.png" alt="paragraph example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/paragraph_4.png" alt="paragraph example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/design_1.png" alt="design example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/design_2.png" alt="design example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/design_3.png" alt="design example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/design_4.png" alt="design example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/scene_1.png" alt="scene example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/scene_2.png" alt="scene example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/scene_3.png" alt="scene example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/scene_4.png" alt="scene example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/cn_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/cn_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/cn_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/cn_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/fr_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/fr_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/fr_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/fr_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/de_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/de_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/de_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/de_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/jp_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/jp_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/jp_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/jp_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/kr_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/kr_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/kr_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/kr_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/es_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/es_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/es_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/es_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/it_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/it_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/it_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/it_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/pt_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/pt_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/pt_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/pt_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
  <tr>
    <td><img src="inference/assets/teaser/ru_1.png" alt="multilingual example 1" width="200"/></td>
    <td><img src="inference/assets/teaser/ru_2.png" alt="multilingual example 2" width="200"/></td>
    <td><img src="inference/assets/teaser/ru_3.png" alt="multilingual example 3" width="200"/></td>
    <td><img src="inference/assets/teaser/ru_4.png" alt="multilingual example 4" width="200"/></td>
  </tr>
</table>


## :wrench: Usage

For a detailed guide on Glyph-SDXL and Glyph-SDXL-v2 inference, see [this folder](inference/).

For a detailed guide on Glyph-ByT5 alignment pretraining, see [this folder](pretraining/).

## :mailbox_with_mail: Citation
If you find this code useful in your research, please consider citing:

```
@article{liu2024glyph,
  title={Glyph-byt5: A customized text encoder for accurate visual text rendering},
  author={Liu, Zeyu and Liang, Weicong and Liang, Zhanhao and Luo, Chong and Li, Ji and Huang, Gao and Yuan, Yuhui},
  journal={arXiv preprint arXiv:2403.09622},
  year={2024}
}
```

and 

```
@article{liu2024glyphv2,
  title={Glyph-ByT5-v2: A Strong Aesthetic Baseline for Accurate Multilingual Visual Text Rendering},
  author={Liu, Zeyu and Liang, Weicong and Zhao, Yiming and Chen, Bohan and Li, Ji and Yuan, Yuhui},
  journal={arXiv preprint arXiv:2406.10208},
  year={2024}
}
```
