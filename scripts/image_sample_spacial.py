"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
os.environ['OPENAI_LOGDIR'] = './logs'

import numpy as np
import torch as th
import torch.distributed as dist
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from cm.image_datasets import load_data

from cm import dist_util, logger
from cm.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample, karras_sample_spacial

def seed_everything(seed):
    """
    Set the seed for all random number generators to ensure reproducibility.
    """
    th.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

def main():
    args = create_argparser().parse_args()
    # print(args.class_cond)
    seed_everything(0)
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        random_flip=False,
        deterministic=False
    )
    
    dist_util.setup_dist()
    logger.configure(dir="sample")

    if "consistency" in args.training_mode:
        distillation = True
    else:
        distillation = False

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        distillation=distillation,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    # if args.sampler == "multistep":
    #     assert len(args.ts) > 0
    #     ts = tuple(int(x) for x in args.ts.split(","))
    # else:
    #     ts = None
    ts = None


    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)

    # while len(all_images) * args.batch_size < args.num_samples:
    # model_kwargs = {}
    # if args.class_cond:
    #     classes = th.randint(
    #         low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
    #     )
    #     model_kwargs["y"] = classes

    imgs, cond = next(data)
    print(cond)
    imgs = imgs.cuda()
    model_kwargs = {
        k: v.cuda()
        for k, v in cond.items()
    }
    
    sample, input_noised, noise_weight = karras_sample_spacial(
        diffusion,
        model,
        imgs,
        (args.batch_size, 3, args.image_size, args.image_size),
        steps=args.steps,
        model_kwargs=model_kwargs,
        device=dist_util.dev(),
        clip_denoised=args.clip_denoised,
        sampler=args.sampler,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        s_churn=args.s_churn,
        s_tmin=args.s_tmin,
        s_tmax=args.s_tmax,
        s_noise=args.s_noise,
        generator=generator,
        ts=ts,
        block_size=args.block_size,
        step_num=args.step_num,
        light_noise_indice=args.light_noise_indice,
        heavy_noise_indice=args.heavy_noise_indice,
    )
    sample = ((sample + 1) / 2)
    input_noised = ((input_noised + 1) / 2)
    imgs = ((imgs + 1) / 2)
    # sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    save_image(
        sample,
        os.path.join(
            f"sample_spacial_{args.sampler}.png"
        ),
    )  # Save the sample as a single image for testing purposes
    save_image(
        input_noised,
        os.path.join(
            f"input_noised_spacial_{args.sampler}.png"
        ),
    )
    save_image(
        noise_weight,
        os.path.join(
            f"noise_weight_spacial_{args.sampler}.png"
        ),
    )
    save_image(
        imgs,
        os.path.join(
            f"imgs_gt_spacial_{args.sampler}.png"
        ),
    )
    # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
    # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
    # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
    # if args.class_cond:
    #     gathered_labels = [
    #         th.zeros_like(classes) for _ in range(dist.get_world_size())
    #     ]
    #     dist.all_gather(gathered_labels, classes)
    #     all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
    # logger.log(f"created {len(all_images) * args.batch_size} samples")

    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # if args.class_cond:
    #     label_arr = np.concatenate(all_labels, axis=0)
    #     label_arr = label_arr[: args.num_samples]
    # for idx in range(len(all_images)):
    #     plt.imsave(
    #         os.path.join(f"sample_{idx}.png"),
    #         all_images[idx],
    #     )

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
        data_dir="",
        block_size=16,
        step_num=4,
        class_cond=False,
        light_noise_indice=30,
        heavy_noise_indice=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
