import torch
import torch.nn.functional as F


def _gradient_xy(image):
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return dx, dy


def gradient_loss(fused, mri, ct):
    """Encourage fused gradients to keep the strongest source edges."""
    fused_dx, fused_dy = _gradient_xy(fused)
    mri_dx, mri_dy = _gradient_xy(mri)
    ct_dx, ct_dy = _gradient_xy(ct)

    target_dx = torch.maximum(torch.abs(mri_dx), torch.abs(ct_dx))
    target_dy = torch.maximum(torch.abs(mri_dy), torch.abs(ct_dy))

    return F.l1_loss(torch.abs(fused_dx), target_dx) + F.l1_loss(torch.abs(fused_dy), target_dy)


def intensity_loss(fused, mri, ct):
    """Keep fused intensity close to the locally stronger source signal."""
    target = torch.maximum(mri, ct)
    return F.l1_loss(fused, target)


def fusion_loss(fused, mri, ct, gradient_weight=1.0, intensity_weight=1.0):
    return (
        intensity_weight * intensity_loss(fused, mri, ct)
        + gradient_weight * gradient_loss(fused, mri, ct)
    )


def gan_generator_loss(fake_logits):
    """Non-saturating generator GAN loss."""
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def gan_discriminator_loss(real_logits, fake_logits):
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (real_loss + fake_loss)


def total_generator_loss(fused, mri, ct, fake_logits=None, fusion_weight=1.0, gan_weight=0.01):
    base_loss = fusion_loss(fused, mri, ct)
    if fake_logits is None:
        gan_loss = fused.new_tensor(0.0)
    else:
        gan_loss = gan_generator_loss(fake_logits)
    return fusion_weight * base_loss + gan_weight * gan_loss
