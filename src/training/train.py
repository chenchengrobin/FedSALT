import gc
from src.config import *

def compute_accuracy(logits, targets):
    labels = targets.argmax(dim=1) if targets.dim() > 1 else targets
    return (logits.argmax(dim=1) == labels).sum().item()


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    clip_grad=None,
):
    model.train()
    total_loss, correct, total_samples = 0.0, 0, 0

    for batch in dataloader:
        data, targets = batch[:2]
        data = data.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        logits = model(data, return_feature=False)
        loss = criterion(logits, targets, *batch[2:])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        bs = data.size(0)
        total_samples += bs
        total_loss += loss.item() * bs
        correct += compute_accuracy(logits, targets)

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * correct / total_samples
    return avg_loss, avg_acc


@torch.inference_mode()
def validate_one_epoch(model, dataloader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss, correct, total_samples = 0.0, 0, 0

    for batch in dataloader:
        data, targets = batch[:2]
        data = data.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        logits = model(data, return_feature=False)
        loss = criterion(logits, targets)

        bs = data.size(0)
        total_samples += bs
        total_loss += loss.item() * bs
        correct += compute_accuracy(logits, targets)

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * correct / total_samples
    return avg_loss, avg_acc


def train_model(
    model,
    optimizer,
    data_loader,
    valid_dataloader=None,
    criterion=None,
    clip_grad=None,
    epochs=200,
    early_stop_patience=10,
    verbose=False,
    need_return=True,
    scheduler=None
):
    best_params = model.state_dict()
    best_loss = float('inf')
    early_stop_counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=data_loader,
            optimizer=optimizer,
            criterion=criterion,
            clip_grad=clip_grad
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        if valid_dataloader is not None:
            val_loss, val_acc = validate_one_epoch(
                model=model,
                dataloader=valid_dataloader
            )

            history['valid_loss'].append(val_loss)
            history['valid_acc'].append(val_acc)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_patience and early_stop_counter >= early_stop_patience:
                    if verbose:
                        print(
                            f"[Early Stop] Epoch {epoch}, "
                            f"best val loss = {best_loss:.4f}"
                        )
                    break
        else:
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(train_loss)
                else:
                    scheduler.step()
            if train_loss < best_loss:
                best_loss = train_loss
                best_params = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_patience and early_stop_counter >= early_stop_patience:
                    if verbose:
                        print(f"[Early Stop] Epoch {epoch}, best train loss = {best_loss:.4f}")
                    break

        if verbose:
            msg = f"Epoch {epoch:3d} | Train Acc {train_acc:.2f}% | Train Loss {train_loss:.4f}"
            if valid_dataloader is not None:
                msg += f" | Val Acc {val_acc:.2f}% | Val Loss {val_loss:.4f}"
            print(msg)

    model.load_state_dict(best_params)
    torch.cuda.empty_cache()
    gc.collect()

    if need_return:
        return history['train_acc']
