# Fix vLLM Compatibility Issue

## Description
This pull request addresses the compatibility issue between TeleChat2 models and vLLM (issue #52 and issue #18). The problem occurs when trying to load TeleChat2 models in vLLM, resulting in the error: `TypeError: _init_model() got an unexpected keyword argument 'layer_type'`.

## Root Cause
The issue stems from differences in model configuration parameters between TeleChat2 and what vLLM expects. Specifically:
- TeleChat2 uses custom parameter names like `n_head` and `n_layer` instead of standard names
- vLLM's model loading logic doesn't recognize TeleChat2's configuration format
- Missing adapter logic to map TeleChat2 parameters to vLLM-compatible parameters

## Solution
This PR provides a comprehensive guide and patch files to enable TeleChat2 models to work with vLLM:

1. Added documentation explaining the integration process
2. Created patch files to modify vLLM for TeleChat2 support
3. Implemented parameter mapping logic to translate between TeleChat2 and vLLM configurations

## Files Added
- `vllm_integration_guide.md`: Detailed guide on integrating TeleChat2 with vLLM
- `vllm_config_patch.diff`: Patch file for modifying vLLM configuration handling
- `telechat_vllm_adapter.py`: Adapter code for seamless integration

## Testing
The solution has been tested with TeleChat2-7B model and verified to resolve the original error. Users can now run TeleChat2 models with vLLM using the standard API:

```python
from vllm import LLM

llm = LLM(model="TeleChat/TeleChat2-7B", trust_remote_code=True)
```

## Additional Notes
This fix maintains backward compatibility and follows vLLM's recommended approach for adding new model support. The changes are minimal and focused specifically on the configuration mapping issue.