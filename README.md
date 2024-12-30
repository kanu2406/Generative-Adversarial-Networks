# Gaussian Mixture Generative Adversarial Networks (GM-GAN)

## Overview

This repository contains the implementation and experiments for **Gaussian Mixture Generative Adversarial Networks (GM-GANs)**, an approach designed to improve the quality and diversity of generated samples by leveraging advanced latent space modeling techniques.Classic Generative Adversarial Networks (GANs) have key limitations, such as mode collapse, where the generator produces limited output varieties, leading to poor
generalization and a lack of diversity. This makes GANs unreliable for tasks requiring broad variation. Additionally, classic GANs lack a mechanism to ensure generated
samples align with specific data classes, hindering control and interpretation, especially in class-conditional generation tasks. These challenges have driven research
into various GAN extensions aimed at overcoming these issues. This report summarizes our work on improving image generation quality and diversity in Generative
Adversarial Networks (GANs) through advanced latent space modeling techniques.

We experimented with different variations of Gaussian Mixture GANs (GM-GANs), including static models, static models enhanced with Wasserstein loss, static models
with an Expectation Maximization approach where we reconstructed the latent space by inverting the generation process with the real images, and a dynamic model
with supervision, which ultimately proved to be the best-performing model. With this last approach, we achieved measurable improvements in image quality and
diversity, as evidenced by enhanced Frechet Inception Distance (FID), precision, and recall metrics. To further investigate the effectiveness of our approach, we also used ´
visualizations of the generated images, confirming better inter-class diversity and quality within each class, highlighting the interpretive benefits of using supervised
GM-GANs for class-distinct and high-fidelity samples.


## Team Members
- **Kanupriya Jain**
- **Alexandre Olech**
- **Lucas Rousselet**





