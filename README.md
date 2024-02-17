# Aligning Modalities in Vision Large Language Models via Preference Fine-tuning

[Yiyang Zhou*](https://yiyangzhou.github.io/), [Chenhang Cui*](https://gzcch.github.io/), [Rafael Rafailov](https://rmrafailov.github.io/), [Chelsea Finn](https://ai.stanford.edu/~cbfinn/), [Huaxiu Yao](https://www.huaxiuyao.io/)
<div align="center">
*Equal Contribution
</div>
<div align="center">
    <a href="https://arxiv.org"><img src="assets/Paper-Arxiv-orange.svg" ></a>
</div>

## News
* 🔥 [2.17] Our paper is online now: https://arxiv.org.

## Getting Started
### Installation

**1. Prepare the code and the environment**
```bash
git clone https://github.com/YiyangZhou/POVID.git
cd POVID
conda create -n llava python=3.10 -y
conda activate POVID
pip install --upgrade pip
pip install -e .
pip install trl
```

**2. Prepare the weights of two stages**
**(Step 1)**: Modify the model preference through DPO (Direct Preference Optimization).
**(Step 2)**: Mitigating Inherent Hallucination Patterns.
|                                The first stage checkpoint 7B (Merged)                               |                                The second stage checkpoint (LoRa)                               |
:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
[Hugingface](https://huggingface.co/YiyangAiLab/) | [Hugingface](https://huggingface.co/YiyangAiLab/)

**3.Inference**
After you have prepared your images and instruction data, you can reason with the following code.
```
python povid_infer.py --model-path [Path to the second stage checkpoint] --model-base [Path to the first stage checkpoint] --input_dir [Path to the images]  --output_file [Path to the output_file]
```

### How to train your own model?


## Related Projects

- [DPO](https://github.com/eric-mitchell/direct-preference-optimization)
- [CHAIR](https://github.com/LisaAnne/Hallucination)
- [Vicuna](https://github.com/lm-sys/FastChat)
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL)
- [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl)
- [LLaVA 1.5](https://github.com/haotian-liu/LLaVA)
- [VLFeedback](https://github.com/vlf-silkie/VLFeedback)
- [Bingo](https://github.com/gzcch/Bingo)

## Citation
If you found this work useful, consider giving this repository a star and citing our paper as followed:
```
@article{zhou2023analyzing,
  title={Analyzing and mitigating object hallucination in large vision-language models},
  author={Zhou, Yiyang and Cui, Chenhang and Yoon, Jaehong and Zhang, Linjun and Deng, Zhun and Finn, Chelsea and Bansal, Mohit and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2310.00754},
  year={2023}
}

@article{cui2023holistic,
  title={Holistic analysis of hallucination in gpt-4v (ision): Bias and interference challenges},
  author={Cui, Chenhang and Zhou, Yiyang and Yang, Xinyu and Wu, Shirley and Zhang, Linjun and Zou, James and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2311.03287},
  year={2023}
}
```
