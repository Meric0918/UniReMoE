"""Keep Full supervision and directed consistency; adapt balance to the expert count."""
from .stage_final_loss import compute_stage_final_loss, LossStatistics


def compute_stage_standard_moe_loss(output, labels, masks, *, task, settings, epoch):
    result = compute_stage_final_loss(output, labels, masks, task=task, settings=settings, epoch=epoch)
    # The Full objective's 0.25 target is exactly correct for the default four experts.
    if output.routing[task].shape[-1] != 4 and result.valid_labels:
        balance = (output.routing[task].float().mean(0) - 1 / output.routing[task].shape[-1]).square().sum()
        result.total = result.total + settings.balance_lambda * (balance - result.balance)
        result.balance = balance
        result.components.update(balance=balance, total=result.total)
    return result
