import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
import wandb
import numbers


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    if num_workers > 0:
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers, pin_memory=True, persistent_workers=True)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    if accelerator.is_main_process:
        wandb.init(
            project="vace-sft",
            name=args.run_name if args and hasattr(args, "run_name") else None,
            config={
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            }
        )
    global_step = 0

    for epoch_id in range(num_epochs):
        pbar = tqdm(dataloader, disable=not accelerator.is_main_process)
        for data in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    model_output = model({}, inputs=data)
                else:
                    model_output = model(data)

                metrics = {}
                if isinstance(model_output, tuple) and len(model_output) == 2 and isinstance(model_output[1], dict):
                    loss, metrics = model_output
                else:
                    loss = model_output

                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()

            global_step += 1
            if accelerator.sync_gradients and accelerator.is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                log_data = {
                    "train/lr": lr,
                    "epoch": epoch_id,
                }

                for k, v in metrics.items():
                    if hasattr(v, "item"):
                        val = v.item()
                    elif isinstance(v, numbers.Number):
                        val = float(v)
                    else:
                        continue
                    log_data[f"train/{k}"] = val

                wandb.log(log_data, step=global_step)

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{lr:.2e}"
                )

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)

    if accelerator.is_main_process:
        wandb.finish()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
