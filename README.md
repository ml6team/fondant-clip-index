# Building a Datacomp CLIP index with Fondant

<p align="center">
    <a href="https://github.com/ml6team/fondant">
        <img src="https://raw.githubusercontent.com/ml6team/fondant/main/docs/art/fondant_banner.svg" height="150px"/>
    </a>
</p>

<p align="center">
    <i>
        <b>Production-ready</b> 
        data processing made 
        <b>easy</b> 
        and 
        <b>shareable</b>
    </i>
    <br>
    <a href="http://fondant.ai"><strong>Explore the Fondant docs »</strong></a>
    <br>
    <br>
    <a href="https://discord.gg/HnTdWhydGp"><img alt="Discord" src="https://dcbadge.vercel.app/api/server/HnTdWhydGp?style=flat-square"></a>
    <a href="https://fondant.readthedocs.io/en/latest/license/"><img alt="License" src="https://img.shields.io/github/license/ml6team/fondant?style=flat-square&color=brightgreen"></a>
</p>

## Introduction

This repository contains the code to build a CLIP index for the Datacomp-12.8M dataset with 
Fondant. It should be straightforward to apply it to a different dataset.

The resulting index has been published on the Hugging Face Hub [here](link). Continue reading 
below to learn:
- [Why we need a CLIP index](#why-a-clip-index)
- [How to use the CLIP index](#using-the-index)
- [Which steps are needed to create the index](#creating-the-index)
- [The execution details of our run](#execution-details)
- [What's next](#whats-next)

## Why a CLIP index?

Large (image) datasets are often unwieldy to use due to their sheer size. Filtering them down to
a useful subset of specific images is expensive if you need to look at every image. Especially
if this is done over and over again for different use cases. Instead, we can look at every image
once, and calculate a (CLIP) embedding representing its content. Combining these embeddings into
an index, we can search through the dataset with a query, finding specific images without
having to look at each one.

![CLIP index](docs/art/clip_index.png)

This is what LAION did for their [LAION-5b dataset](https://laion.ai/blog/laion-5b/), which made 
it possible to use, like we did in our 
[ControlNet example](https://github.com/ml6team/fondant-usecase-controlnet). 
Unfortunately, the LAION-5b dataset and index have been 
[taken offline](https://laion.ai/notes/laion-maintanence/) (temporarily) and there 
[aren't any alternatives](https://github.com/rom1504/clip-retrieval/issues/324). This is
why we built an index for the Datacomp-12M dataset. While it is a lot smaller than LAION-5b, it
should already enable a lot of use cases again, and can hopefully be the start towards building
indices for more and larger datasets.

## Creating the index

We leveraged Fondant to generate the CLIP index and published the pipeline in this git 
repository. You can find it in [`pipeline.py`](pipeline.py).
The pipeline consists of 4 steps:

- A [`load_from_hf_hub`](https://fondant.ai/en/stable/components/hub/#load_from_hf_hub#description) 
  operation that loads the 
  [datacomp_small](https://huggingface.co/datasets/mlfoundations/datacomp_small) dataset from 
  huggingface into the Fondant workspace and format.
- A [`download_images`](https://fondant.ai/en/stable/components/hub/#download_images#description)
  operation which downloads the actual images from the urls in the dataset.
- A [`embed_images`](https://fondant.ai/en/stable/components/hub/#embed_images#description) operation which embeds the downloaded images using a CLIP model.
- A [`write_to_file`](https://fondant.ai/en/stable/components/hub/#write_to_file#description) 
  operation which writes the original urls and generated embeddings to the chosen destination.

After running the pipeline, we used [`autofaiss`](https://github.com/criteo/autofaiss) to build the 
CLIP index. You can use the included wrapper script [`build_index.py`](build_index.py).

Once you have created the index, you can explore your index and validate that everything is 
working using the [`exploration.ipynb`](exploration.ipynb) notebook.

## Using the index

### With Fondant

The easiest way to use the index, is using Fondant. Fondant offers reusable operations which
allow you to query the index with your data, either prompts or images:
- link
- link

To see how it can be used in an end-to-end example, check our 
[ControlNet example](https://github.com/ml6team/fondant-usecase-controlnet) which
uses the index to create a dataset to fine-tune a ControlNet model on a specific domain.

### With Clip-Retrieval

There are other open source tools which allow you to leverage a CLIP index. We can recommend
[clip-retrieval](https://github.com/rom1504/clip-retrieval) which lets you set up a service 
hosting the index accessible by API.

## Execution details

### Download images

We downloaded the images with 32 cores in parallel, each opening up to 25 concurrent connections,
and achieved a success rate of 72%, resulting in 9.251.172 images.

The downloading was executed on a VM on GCP using the Fondant Docker runner. We originally 
planned to run this on Vertex AI, but moved to a VM when noticing lower network bandwidth on Vertex.

The success rate can probably be further improved by setting up a faster DNS resolver.

### Embed images

We leveraged the 
[`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) 
CLIP model. We chose this model because of a couple of reasons. It is popular, which makes it 
easy to use with existing embeddings, it is small, which makes it cheap to run, and it is an open 
model trained on open data.

We appreciate any feedback on our choice of model, so we can take this into account if we
generate indices for larger datasets in the future.

The embedding was executed on 4 T4 GPUs on Google Cloud using our Vertex AI runner, with a batch
size of 32. The execution took 8:15 hours.

## What's next

### Making data building collaborative

With Fondant we aim to make data building collaborative, and we will share more features built 
on top of the Datacomp datasets to showcase this in the future. To stay up to date, join our
[Discord](https://discord.gg/HnTdWhydGp).

### Larger datasets

Based on the popularity and feedback we receive on this 12.8M index, we might generate a CLIP
index for the datacomp-128M dataset. If there are other datasets you are interested in, or want 
to generate an index for a different dataset yourself, please let us know in our 
[Discord](https://discord.gg/HnTdWhydGp).
