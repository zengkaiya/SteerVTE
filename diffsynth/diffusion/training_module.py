import torch, json, os
from ..core import ModelConfig, load_state_dict
from ..utils.controlnet import ControlNetInput
from peft import LoraConfig, inject_adapter_in_model


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        if isinstance(target_modules, list) and len(target_modules) == 1:
            target_modules = target_modules[0]
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        # print(f"Trainable param names: len={len(trainable_param_names)}")
        state_dict_export = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        # Always keep OCR projection weights if present, so OCR linear can be resumed from checkpoints.
        extra_prefixes = (
            "pipe.ocr_image_encoder.proj.",
            "pipe.style_image_encoder.style_proj_out.",
            "pipe.style_image_encoder.proj.",
        )
        for name, param in state_dict.items():
            if any(name.startswith(prefix) for prefix in extra_prefixes):
                state_dict_export[name] = param
        state_dict = state_dict_export
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        if data is None:
            return data
        elif isinstance(data, torch.Tensor):
            data = data.to(device)
            if torch_float_dtype is not None and data.dtype in [torch.float, torch.float16, torch.bfloat16]:
                data = data.to(torch_float_dtype)
            return data
        elif isinstance(data, tuple):
            data = tuple(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, list):
            data = list(self.transfer_data_to_device(x, device, torch_float_dtype) for x in data)
            return data
        elif isinstance(data, dict):
            data = {i: self.transfer_data_to_device(data[i], device, torch_float_dtype) for i in data}
            return data
        else:
            return data
    
    def parse_vram_config(self, fp8=False, offload=False, device="cpu"):
        if fp8:
            return {
                "offload_dtype": torch.float8_e4m3fn,
                "offload_device": device,
                "onload_dtype": torch.float8_e4m3fn,
                "onload_device": device,
                "preparing_dtype": torch.float8_e4m3fn,
                "preparing_device": device,
                "computation_dtype": torch.bfloat16,
                "computation_device": device,
            }
        elif offload:
            return {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": "disk",
                "onload_device": "disk",
                "preparing_dtype": torch.bfloat16,
                "preparing_device": device,
                "computation_dtype": torch.bfloat16,
                "computation_device": device,
                "clear_parameters": True,
            }
        else:
            return {}
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, fp8_models=None, offload_models=None, device="cpu"):
        fp8_models = [] if fp8_models is None else fp8_models.split(",")
        offload_models = [] if offload_models is None else offload_models.split(",")
        model_configs = []
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            for path in model_paths:
                vram_config = self.parse_vram_config(
                    fp8=path in fp8_models,
                    offload=path in offload_models,
                    device=device
                )
                model_configs.append(ModelConfig(path=path, **vram_config))
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            for model_id_with_origin_path in model_id_with_origin_paths:
                vram_config = self.parse_vram_config(
                    fp8=model_id_with_origin_path in fp8_models,
                    offload=model_id_with_origin_path in offload_models,
                    device=device
                )
                config = self.parse_path_or_model_id(model_id_with_origin_path)
                model_configs.append(ModelConfig(model_id=config.model_id, origin_file_pattern=config.origin_file_pattern, **vram_config))
        return model_configs
    

    def parse_path_or_model_id(self, model_id_with_origin_path, default_value=None):
        if model_id_with_origin_path is None:
            return default_value
        elif os.path.exists(model_id_with_origin_path):
            return ModelConfig(path=model_id_with_origin_path)
        else:
            if ":" not in model_id_with_origin_path:
                raise ValueError(f"Failed to parse model config: {model_id_with_origin_path}. This is neither a valid path nor in the format of `model_id/origin_file_pattern`.")
            split_id = model_id_with_origin_path.rfind(":")
            model_id = model_id_with_origin_path[:split_id]
            origin_file_pattern = model_id_with_origin_path[split_id + 1:]
            return ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern)


    def auto_detect_lora_target_modules(
        self,
        model: torch.nn.Module,
        search_for_linear=False,
        linear_detector=lambda x: min(x.weight.shape) >= 512,
        block_list_detector=lambda x: isinstance(x, torch.nn.ModuleList) and len(x) > 1,
        name_prefix="",
    ):
        lora_target_modules = []
        if search_for_linear:
            for name, module in model.named_modules():
                module_name = name_prefix + ["", "."][name_prefix != ""] + name
                if isinstance(module, torch.nn.Linear) and linear_detector(module):
                    lora_target_modules.append(module_name)
        else:
            for name, module in model.named_children():
                module_name = name_prefix + ["", "."][name_prefix != ""] + name
                lora_target_modules += self.auto_detect_lora_target_modules(
                    module,
                    search_for_linear=block_list_detector(module),
                    linear_detector=linear_detector,
                    block_list_detector=block_list_detector,
                    name_prefix=module_name,
                )
        return lora_target_modules
    

    def parse_lora_target_modules(self, model, lora_target_modules):
        if lora_target_modules == "":
            print("No LoRA target modules specified. The framework will automatically search for them.")
            lora_target_modules = self.auto_detect_lora_target_modules(model)
            print(f"LoRA will be patched at {lora_target_modules}.")
        else:
            lora_target_modules = lora_target_modules.split(",")
        return lora_target_modules


    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        task="sft",
    ):
        ckpt_state_dict = None
        if lora_checkpoint is not None and not task.endswith(":data_process"):
            ckpt_state_dict = load_state_dict(lora_checkpoint)

        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Preset LoRA
        if preset_lora_path is not None:
            pipe.load_lora(getattr(pipe, preset_lora_model), preset_lora_path)
        
        # train ocr_image_encoder proj
        if hasattr(pipe, "ocr_image_encoder") and pipe.ocr_image_encoder is not None:
            pipe.ocr_image_encoder.eval()
            pipe.ocr_image_encoder.requires_grad_(False)
            module = getattr(pipe.ocr_image_encoder, "proj", None)
            if module is not None:
                module.train()
                module.requires_grad_(True)

        # train style_image_encoder proj
        if hasattr(pipe, "style_image_encoder") and pipe.style_image_encoder is not None:
            pipe.style_image_encoder.eval()
            pipe.style_image_encoder.requires_grad_(False)
            module = self._get_style_proj_module(pipe.style_image_encoder)
            if module is not None:
                module.train()
                module.requires_grad_(True)

        # FP8
        # FP8 relies on a model-specific memory management scheme.
        # It is delegated to the subclass.
        
        # Add LoRA to the base models
        if lora_base_model is not None and not task.endswith(":data_process"):
            if (not hasattr(pipe, lora_base_model)) or getattr(pipe, lora_base_model) is None:
                print(f"No {lora_base_model} models in the pipeline. We cannot patch LoRA on the model. If this occurs during the data processing stage, it is normal.")
                return
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=self.parse_lora_target_modules(getattr(pipe, lora_base_model), lora_target_modules),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if ckpt_state_dict is not None:
                state_dict = self.mapping_lora_state_dict(ckpt_state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)

        if ckpt_state_dict is not None:
            self._load_ocr_proj_from_checkpoint(pipe, ckpt_state_dict, lora_checkpoint)
            self._load_style_proj_from_checkpoint(pipe, ckpt_state_dict, lora_checkpoint)


    def _load_ocr_proj_from_checkpoint(self, pipe, state_dict, checkpoint_path):
        if (not hasattr(pipe, "ocr_image_encoder")) or pipe.ocr_image_encoder is None:
            return
        if (not hasattr(pipe.ocr_image_encoder, "proj")) or pipe.ocr_image_encoder.proj is None:
            return

        proj_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("pipe.ocr_image_encoder.proj."):
                proj_state_dict[key[len("pipe.ocr_image_encoder.proj."):]] = value
            elif key.startswith("ocr_image_encoder.proj."):
                proj_state_dict[key[len("ocr_image_encoder.proj."):]] = value

        if len(proj_state_dict) == 0:
            return

        load_result = pipe.ocr_image_encoder.proj.load_state_dict(proj_state_dict, strict=False)
        print(f"OCR proj checkpoint loaded: {checkpoint_path}, total {len(proj_state_dict)} keys")
        if len(load_result[1]) > 0:
            print(f"Warning, OCR proj key mismatch! Unexpected keys in checkpoint: {load_result[1]}")


    def _get_style_proj_module(self, style_image_encoder):
        module = getattr(style_image_encoder, "style_proj_out", None)
        if module is None:
            module = getattr(style_image_encoder, "proj", None)
        return module


    def _load_style_proj_from_checkpoint(self, pipe, state_dict, checkpoint_path):
        if (not hasattr(pipe, "style_image_encoder")) or pipe.style_image_encoder is None:
            return
        module = self._get_style_proj_module(pipe.style_image_encoder)
        if module is None:
            return

        proj_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("pipe.style_image_encoder.style_proj_out."):
                proj_state_dict[key[len("pipe.style_image_encoder.style_proj_out."):]] = value
            elif key.startswith("style_image_encoder.style_proj_out."):
                proj_state_dict[key[len("style_image_encoder.style_proj_out."):]] = value
            elif key.startswith("pipe.style_image_encoder.proj."):
                proj_state_dict[key[len("pipe.style_image_encoder.proj."):]] = value
            elif key.startswith("style_image_encoder.proj."):
                proj_state_dict[key[len("style_image_encoder.proj."):]] = value

        if len(proj_state_dict) == 0:
            return

        load_result = module.load_state_dict(proj_state_dict, strict=False)
        print(f"Style proj checkpoint loaded: {checkpoint_path}, total {len(proj_state_dict)} keys")
        if len(load_result[1]) > 0:
            print(f"Warning, Style proj key mismatch! Unexpected keys in checkpoint: {load_result[1]}")


    def split_pipeline_units(self, task, pipe, trainable_models=None, lora_base_model=None):
        models_require_backward = []
        if trainable_models is not None:
            models_require_backward += trainable_models.split(",")
        if lora_base_model is not None:
            models_require_backward += [lora_base_model]
        # OCR proj is toggled in `switch_pipe_to_training_mode`, but unit split happens earlier.
        # If OCR encoder exists, keep its unit in the train graph so proj can receive gradients.
        if hasattr(pipe, "ocr_image_encoder") and pipe.ocr_image_encoder is not None:
            models_require_backward += ["ocr_image_encoder"]
        if hasattr(pipe, "style_image_encoder") and pipe.style_image_encoder is not None:
            models_require_backward += ["style_image_encoder"]
        # `freeze_except` supports nested names like `ocr_image_encoder.proj`,
        # but unit graph matching relies on top-level `onload_model_names`.
        models_require_backward += [name.split(".")[0] for name in models_require_backward if name != ""]
        models_require_backward = sorted(list(set(models_require_backward)))
        if task.endswith(":data_process"):
            _, pipe.units = pipe.split_pipeline_units(models_require_backward)
        elif task.endswith(":train"):
            pipe.units, _ = pipe.split_pipeline_units(models_require_backward)
        return pipe
    
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        controlnet_keys_map = (
            ("blockwise_controlnet_", "blockwise_controlnet_inputs",),
            ("controlnet_", "controlnet_inputs"),
        )
        controlnet_inputs = {}
        for extra_input in extra_inputs:
            for prefix, name in controlnet_keys_map:
                if extra_input.startswith(prefix):
                    if name not in controlnet_inputs:
                        controlnet_inputs[name] = {}
                    controlnet_inputs[name][extra_input.replace(prefix, "")] = data[extra_input]
                    break
            else:
                inputs_shared[extra_input] = data[extra_input]
        for name, params in controlnet_inputs.items():
            inputs_shared[name] = [ControlNetInput(**params)]
        return inputs_shared
